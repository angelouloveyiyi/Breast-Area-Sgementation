import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
import keras

'''
img = np.load('D:\BC/image_seg.npy')
one_img = img[2:3, :, :, :]
img_in = (np.squeeze(one_img) + 0.5) * 255
plt.figure()
plt.imshow(img_in, cmap='gray')

'''

with open("D:\\Breast Region by DL\\train_hist.txt", 'r') as f:
    data = json.load(f)
    loss = data['loss']

plt.figure()
plt.plot(loss)
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.savefig("D:\\Breast Region by DL\\MSE.png")

segNN = keras.models.load_model("D:\\Breast Region by DL\\segNN.h5")

# read original images and pre-process
org_img = np.load('D:\\Breast Region by DL\\test_image.npy')
x_train = org_img.astype('float32') / 255. - 0.5       # minmax_normalized
x_train = np.expand_dims(x_train, axis=3)
for i in range(x_train.shape[0]):
    # encode one img
    one_img = x_train[i:i+1, :, :, :]
    img_in = (np.squeeze(one_img) + 0.5) * 255
    img_in = img_in.transpose(1,0)
    #plt.figure()
    #plt.imshow(img_in, cmap='gray')
    #plt.savefig("ORG_%d.png" % i)
    matplotlib.image.imsave("D:\\Breast Region by DL\\987654\\ORG_%d.png" % i, img_in, cmap='gray')

    img = segNN.predict(one_img)
    img = (np.squeeze(img) + 0.5) * 255
    img = img.transpose(1,0)
    #plt.figure()
    #plt.imshow(img, cmap='gray')
    #plt.savefig("Seg_%d.png" % i)
    matplotlib.image.imsave("D:\\Breast Region by DL\\987654\\Seg_%d.png" % i, img, cmap='gray')
# -----------------------------------------------------------------------