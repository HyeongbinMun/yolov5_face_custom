import argparse
import os

def communicator_video(module_url, video_path, video_text, extract_fps, start_time, end_time, module_type):
    json_video = open(video_path, 'rb')
    json_files = {'video': json_video}
    json_data = dict({
        "analysis_type": module_type,
        "video_text": video_text,
        "extract_fps": extract_fps,
        "start_time": start_time,
        "end_time": end_time
    })
    result_response = requests.post(url=module_url, data=json_data, files=json_files)
    result_data = json.loads(result_response.content)
    result = result_data['result']

    return result

def create_txt(img_path, save_txt, type):
    if type=='image':
        img_list = os.listdir(img_path)
        with open(save_txt, "w") as f:
            for img in img_list:
                f.write(img_path + img + '\n')
    else:
        img_dir_list = os.listdir(img_path)
        with open(save_txt, "w") as f:
            for img_dir in img_dir_list:
                img_dir_path = os.path.join(img_path, img_dir)
                img_list = os.listdir(img_dir_path)
                for img in img_list:
                    f.write(os.path.join(img_dir, img) + '\n')
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='widerface_evaluate/results/frame/', help='save to project/name')
    parser.add_argument('--save_txt', default='widerface_evaluate/results/demo_ver2.txt', help='save to project/name')
    parser.add_argument('--type', default='image', help='save to image or folder')

    opt = parser.parse_args()
    print(opt)

    create_txt(opt.img_dir, opt.save_txt, opt.type)