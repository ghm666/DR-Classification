from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import  train_test_split
import pandas as pd

from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

# 参数设置
img_rows, img_cols = 224, 224
nb_classes = 5
batchsize= 32
train_epoch=30

# 加载数据集
immatrix = np.load("immatrix.npy")
imlabel = np.load("imlabel.npy")
print("immatrix: ",immatrix.shape)
print("imlabel: ",imlabel.shape)

# 显示训练图像
# import matplotlib.pyplot as plt
# for i in range (10):
#     img=immatrix[i].reshape(img_rows,img_cols,3)
#     print('severity',imlabel[i])
#     plt.imshow(img)
#     plt.show()

# 打乱数据
data,Label = shuffle(immatrix,imlabel, random_state=2)
train_data = [data,Label]
(X, y) = (train_data[0],train_data[1])

# 训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
X_train = X_train.reshape(X_train.shape[0], img_cols, img_rows, 3)
X_test = X_test.reshape(X_test.shape[0], img_cols, img_rows, 3)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# 构建模型
vgg16 = VGG16(weights='imagenet')

x = vgg16.get_layer('fc2').output
prediction = Dense(5, activation='softmax', name='predictions')(x)

model = Model(inputs=vgg16.input, outputs=prediction)


for layer in model.layers[:17]:
    layer.trainable = False
for layer in model.layers[17:]:
    layer.trainable = True


# create generators  - training data will be augmented images
validationdatagenerator = ImageDataGenerator()
traindatagenerator = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,rotation_range=15,zoom_range=0.1 )


train_generator=traindatagenerator.flow(X_train, Y_train, batch_size=batchsize)
validation_generator=validationdatagenerator.flow(X_test, Y_test,batch_size=batchsize)

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])
sgd = SGD(lr=1e-3, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

filepath="./model_save/vgg-{epoch:02d}.h5"
checkpointer1 = keras.callbacks.ModelCheckpoint(filepath,
                               monitor='loss',
                               verbose=1,
                               save_best_only=True,
                               save_weights_only=False,
                               mode='auto',
                               period=5)

tensroboad = keras.callbacks.TensorBoard(log_dir='./logs')



history= model.fit_generator(train_generator, steps_per_epoch=int(len(X_train)/batchsize),
                    epochs=train_epoch, validation_data=validation_generator,
                    validation_steps=int(len(X_test)/batchsize),
                    callbacks = [checkpointer1, tensroboad])