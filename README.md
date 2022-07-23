# Face_detection
본 문서는 face_detection에 관한 code입니다

## Settings
1. docker-compose.yml
 - 이름 재정의
    ```
    container_name : rename
    ```

 - 포트 재설정
    ```
    ports:
      - "10000:22"
      - "11000:6006"
      - "12000:8000"
    ```
    
 - docker container 설치
    ```
    docker-compose up -d #git clone dir 이동 후 실행
    ```

2. error report
 - ImportError: libGL.so.1: cannot open shared object file: No such file or directory
 - 해당 모듈 설치
    ```
    apt-get install libgl1-mesa-glx
    ```

## Code
1. create_txt.py
 - 미리 img 경로가 모두 나와있는 txt 생성
 - video의 경우 ffmpeg으로 해당 frame에 맞게 전부 cutting
    ```
    ffmpeg -i test.mp4 -r 30 frame/%d.jpg          # test.mp4를 30 fps로 frame folder에 이미지 cutting
    ```
    ```
    python create_txt.py \
    --img_dir /workspace/data/sengro/frame/ \      # 해당 img directory
    --save_txt /workspace/data/sengro/sengro.txt \ # img_dir txt가 저장되는 경로
    --type image                                   # image or folder 선택(image : img dir에 img만 있는 경우, folder : img dir에 folder마다 img가 있는 경우)
    ```
    
2. test_widerface.py
 - 모든 image에 대한 face position을 txt로 저장
    ```
    python test_widerface.py \
    --weight weights/result_face_l.pt \
    --img-size 640 \
    --dataset_folder img_path \ 이미지만 존재하는 dir
    --folder_pict img_txt \ 이미지 경로 모두 포함된 txt
    --save_folder save_path

    ex)
    python test_widerface.py \
    --weight weights/result_face_l.pt \
    --img-size 640 \
    --dataset_folder data/test/images/ \
    --folder_pict data/test/test_label.txt \
    --save_folder widerface_evaluate/widerface_txt/
    ```
 
 3. draw.py
 - 해당 face position에 따라서 bounding box image 생성
    ```
    python draw.py \
    --img_dir data/test/demo/ \                        # face position을 그리는 img dir
    --img_txt widerface_evaluate/widerface_txt/demo/ \ # img의 face position이 나온 txt, test_widerface로 생성
    --save_dir widerface_evaluate/results/demo/        # face box가 그려진 save img dir
    ```
    
4. video 생성
    ```
    ffmpeg -r 30 -i frame/%d.jpg -vcodec libx264 test.mp4
    # draw.py로 생성된 image로 frame 30의 h264 디코더 형식의 test.mp4 생성
    ```
