from PIL import Image
import numpy as np
import os
import pandas as pd


img_dir = "F:\\DR_detection_dataset\\new_train_test"
listing = os.listdir(img_dir)
print(np.size(listing))
trainLabels = pd.read_csv("F:\\DR_detection_dataset\\trainLabels\\trainLabels.csv")

img_rows, img_cols = 224, 224
immatrix = []
imlabel = []

for file in listing:
    file1 = file.split('.')
    imlabel.append(trainLabels.loc[trainLabels.image== file1[0], 'level'].values[0])
    print(type(trainLabels.loc[trainLabels.image== file1[0], 'level'].values[0]))
    img_path = os.path.join(img_dir, str(file))
    im = Image.open(img_path)
    img = im.resize((img_rows,img_cols))
    rgb = img.convert('RGB')
    immatrix.append(np.array(rgb).flatten())

#converting images & labels to numpy arrays
immatrix = np.asarray(immatrix)
imlabel = np.asarray(imlabel)
np.save("immatrix.npy",immatrix)
np.save("imlabel.npy",imlabel)







