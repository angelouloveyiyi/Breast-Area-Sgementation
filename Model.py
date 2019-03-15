import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras import applications, optimizers, callbacks
from keras.layers import Dense, Input, MaxPooling2D, BatchNormalization, Activation, Reshape, Flatten
from keras.layers.convolutional import UpSampling2D, Conv2D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.layers import MaxPooling2D, UpSampling2D, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import concatenate

img_shape = (256, 128, 1)
epochs = 100 #  s
batch_size = 6


x_train = np.load('D:\Frank\Box Sync\GRP_Loew-Doc\Breast Region by DL\\train_image.npy')  # input data
y_train = np.load('D:\Frank\Box Sync\GRP_Loew-Doc\Breast Region by DL\\train_label.npy')  # input seg
# data pre-processing
x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
x_train = np.expand_dims(x_train, axis=3)

y_train = y_train.astype('float32') / 255. - 0.5       # minmax_normalized
y_train = np.expand_dims(y_train, axis=3)

dropout = 0.2
def build_model(pretrained_weights=None, input_size=img_shape):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    drop1 = Dropout(dropout)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(dropout)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(drop5)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv6)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv7)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same',
                          kernel_initializer='he_normal')(conv8)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    #sgd = optimizers.SGD(lr=0.0001, decay=1e-2, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizers.Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=1e-6), loss='mse')
    #model.compile(optimizer = 'sgd',loss='mse')
    # model.summary()

    if (pretrained_weights): 
        model.load_weights(pretrained_weights)

    return model


# construct the model
segNN = build_model()
#model_checkpoint = ModelCheckpoint('segNN.hdf5', monitor='loss',verbose=1, save_best_only=True)
# compile
# segNN.compile(loss='mse', optimizer=optimizers.Nadam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004))
# training
hist = segNN.fit(x_train, y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True)

segNN.save('D:\Frank\Box Sync\GRP_Loew-Doc\Breast Region by DL\\segNN.h5')

f1 = open('D:\Frank\Box Sync\GRP_Loew-Doc\Breast Region by DL\\train_hist.txt', 'w')
json.dump(hist.history, f1)
f1.close()


'''
img_seg = segNN.predict(x_train)
np.save('D:\BC/image_seg.npy', img_seg)
'''