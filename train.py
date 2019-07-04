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
vgg16_model = VGG16(weights="imagenet", include_top=True)
# visualize layers
print("VGG16 model layers")
for i, layer in enumerate(vgg16_model.layers):
    print(i, layer.name, layer.output_shape)

# (2) remove the top layer
base_model = Model(inputs=vgg16_model.input,
                   outputs=vgg16_model.get_layer("block5_pool").output)
from keras.layers import Dense, Dropout, Reshape

# (3) attach a new top layer
base_out = base_model.output
base_out = Reshape((25088,))(base_out)
top_fc1 = Dense(256, activation="relu")(base_out)
top_fc1 = Dropout(0.5)(top_fc1)
top_preds = Dense(5, activation="softmax")(top_fc1)
# (4) freeze weights until the last but one convolution layer (block4_pool)
for layer in base_model.layers[0:14]:
    layer.trainable = False

# (5) create new hybrid model
model = Model(inputs=base_model.input, outputs=top_preds)


# create generators  - training data will be augmented images
validationdatagenerator = ImageDataGenerator()
traindatagenerator = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,rotation_range=15,zoom_range=0.1 )


train_generator=traindatagenerator.flow(X_train, Y_train, batch_size=batchsize)
validation_generator=validationdatagenerator.flow(X_test, Y_test,batch_size=batchsize)

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])
sgd = SGD(lr=1e-2, momentum=0.9)
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