import argparse
import cv2
import os

def draw_box(img_dir, img_txt, save_dir, type):
    img_list = os.listdir(img_dir)
    txt_list = os.listdir(img_txt)
    boxes = []
            
    for txt in txt_list:
        txt_path = opt.img_txt + txt
        with open(txt_path, mode='r') as f:
            lines = f.readlines()

            for i  in range(len(lines)):
                lines[i]  = lines[i].strip('\n')

            obj = {'img_name':lines[0], 'obj_num':int(lines[1])}

            box_list = []
            for i in range(2, len(lines)):
                box_el = []
                x1, y1, w, h, score = lines[i].split(' ')
                box_el.append(int(x1))
                box_el.append(int(y1))
                box_el.append(int(w))
                box_el.append(int(h))
                box_el.append(float(score))
                box_list.append(box_el)

            obj['box'] = box_list
            boxes.append(obj)

    print('boxes : ', boxes)
    print('img_list : ', img_list)

    for img in img_list:
        img_path = opt.img_dir + img
        img_name, _ = img.split('.')

        image = cv2.imread(img_path)
        copy_image = image.copy()

        obj = next((item for item in boxes if item['img_name'] == img_name), None)
        for i in range(0, obj['obj_num']):
            if obj['box'][i][4] >= 0.5:
                x1 = obj['box'][i][0]
                y1 = obj['box'][i][1]
                x2 = obj['box'][i][0] + obj['box'][i][2]
                y2 = obj['box'][i][1] + obj['box'][i][3]
                cv2.rectangle(copy_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imwrite(save_dir+img, copy_image)

    # if type == 'video':


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='data/test/images/', help='save to project/name')
    parser.add_argument('--img_txt', default='widerface_evaluate/widerface_txt/images/', help='save to project/name')
    parser.add_argument('--save_dir', default='widerface_evaluate/results/', help='save to project/name')
    parser.add_argument('--type', default='image', help='save to type(video/image)')

    opt = parser.parse_args()
    print(opt)

    draw_box(opt.img_dir, opt.img_txt, opt.save_dir, opt.type)

