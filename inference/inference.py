import argparse
import glob
import time
from pathlib import Path

import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression_face, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from tqdm import tqdm

class Face_inference:
    def __init__(self, img_size=640, weights_path='weights/result_face_l.pt', conf_thres=0.02, iou_thres=0.5, device='0',
                 agnostic_nms='store_true',augment='store_true', save_folder='result/', dataset_folder='data/'):
        self.weights_path = weights_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.save_folder = save_folder
        self.dataset_folder = dataset_folder
        self.folder_pict = self.init_data_pict(self.dataset_folder)
        self.results = []

    @staticmethod
    def init_data_pict(dir):
        folder_pict = {}
        img_list = os.listdir(dir)
        line = dir.strip().split('/')
        for img in img_list:
            folder_pict[img] = line[-2]
        return folder_pict

    @staticmethod
    def dynamic_resize(shape, stride=64):
        max_size = max(shape[0], shape[1])
        if max_size % stride != 0:
            max_size = (int(max_size / stride) + 1) * stride
        return max_size

    @staticmethod
    def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        # clip_coords(coords, img0_shape)
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4
        coords[:, 8].clamp_(0, img0_shape[1])  # x5
        coords[:, 9].clamp_(0, img0_shape[0])  # y5
        return coords

    @staticmethod
    def show_results(img, xywh, conf, landmarks, class_num):
        h, w, c = img.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
        y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
        x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
        y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        for i in range(5):
            point_x = int(landmarks[2 * i] * w)
            point_y = int(landmarks[2 * i + 1] * h)
            cv2.circle(img, (point_x, point_y), tl + 1, clors[i], -1)

        tf = max(tl - 1, 1)  # font thickness
        label = str(int(class_num)) + ': ' + str(conf)[:5]
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return img

    def detect(self, model, device, img0):
        stride = int(model.stride.max())  # model stride
        imgsz = self.img_size
        if imgsz <= 0:  # original size
            imgsz = self.dynamic_resize(img0.shape)
        imgsz = check_img_size(imgsz, s=64)  # check img_size
        img = letterbox(img0, imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=self.augment)[0]
        # Apply NMS
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(img0.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        boxes = []
        h, w, c = img0.shape
        if pred is not None:
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
            pred[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], pred[:, 5:15], img0.shape).round()
            for j in range(pred.size()[0]):
                xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
                xywh = xywh.data.cpu().numpy()
                conf = pred[j, 4].cpu().numpy()
                landmarks = (pred[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = pred[j, 15].cpu().numpy()
                x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
                y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
                x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
                y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
                boxes.append([x1, y1, x2 - x1, y2 - y1, conf])
        return boxes

    def main(self):
        # Load model
        device = select_device(self.device)
        model = attempt_load(self.weights_path, map_location=device)  # load FP32 model
        conf_sum = 0.0
        count = 0
        with torch.no_grad():
            # testing dataset
            testset_folder = self.dataset_folder
            print('a : ', tqdm(glob.glob(os.path.join(testset_folder, '*'))))

            for image_path in tqdm(glob.glob(os.path.join(testset_folder, '*'))):
                if image_path.endswith('.txt'):
                    continue
                img0 = cv2.imread(image_path)  # BGR
                if img0 is None:
                    print(f'ignore : {image_path}')
                    continue
                boxes = self.detect(model, device, img0)
                # --------------------------------------------------------------------
                image_name = os.path.basename(image_path)
                txt_name = os.path.splitext(image_name)[0] + ".txt"
                save_name = os.path.join(self.save_folder, self.folder_pict[image_name], txt_name)
                dirname = os.path.dirname(save_name)
                if not os.path.isdir(dirname):
                    os.makedirs(dirname)

                el = {}
                file_name = os.path.basename(save_name)[:-4] + '.jpg'
                el['file_name'] = file_name
                box_list = []

                for box in boxes:
                    tmp = {}
                    tmp['x1'] = box[0]
                    tmp['y1'] = box[1]
                    tmp['w'] = box[2]
                    tmp['h'] = box[3]
                    tmp['score'] = round(float(box[4]), 2)
                    conf_sum += box[4]
                    count += 1
                    box_list.append(tmp)

                el['boxes'] = box_list
                self.results.append(el)

            print('conf avg : ', conf_sum / count)
            print('done.')
            return self.results

if __name__ == '__main__':
    pass


