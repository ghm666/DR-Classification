import shutil
import os

img_dir = "F:\\DR_detection_dataset\\crop_train"
newimg_dir = "F:\\DR_detection_dataset\\4"
label_dir = "F:\\DR_detection_dataset\\trainLabels\\trainLabels.csv"
with open(label_dir, "r") as f:
    lines = f.readlines()
    number = 1
    for line in lines[1:]:
        print(line)
        line = line.split(',')
        if int(line[1]) == 4:
            print(number)
            number += 1
            img_name = line[0] + '.jpeg'
            img_path = os.path.join(img_dir,img_name)
            if not os.path.exists(img_path):
                continue
            newimg_path = os.path.join(newimg_dir, img_name)
            # shutil.copyfile(img_path, newimg_path)
            shutil.move(img_path, newimg_path)




