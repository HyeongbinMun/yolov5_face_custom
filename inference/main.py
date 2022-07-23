from inference import Face_inference

if __name__ == '__main__':
    face = Face_inference(weights_path='weights/result_face_l.pt', save_folder='result/',
                          dataset_folder='data/', img_size=640)
    results = face.main()
    print(results)