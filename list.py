import os

if __name__ == '__main__':
    img_dir = 'data/test/demo/'
    img_list = os.listdir(img_dir)
    f = open('data/test/demo.txt', 'w')
    for img in img_list:
        img_path = img_dir + img + '\n'
        f.write(img_path)

    f.close()
    
