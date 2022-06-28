import keras
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people
import math
from random import random


# salt&pepper noise
def salt_and_pepper(images, p):
    output = np.zeros(images.shape,np.float32)
    thres = 1 - p
    for idx in range(len(images)):
        for i in range(images[idx].shape[0]):
            for j in range(images[idx].shape[1]):
                rdn = random()
                if rdn < p:
                    output[idx][i][j] = 0
                elif rdn > thres:
                    output[idx][i][j] = 1
                else:
                    output[idx][i][j] = images[idx][i][j]
    return output


# gaussian noise
def Gaussian(img):
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape).reshape(img.shape)
    return img + gauss


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    MAX = 1.0
    return 20 * math.log10(MAX / math.sqrt(mse))


dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.5).images / 255
# CNN 예측을 위한 이미지 resize
temp = np.zeros((len(dataset), 52, 40))
for i, d in enumerate(dataset):
    temp[i] = cv2.resize(d, (40, 52), interpolation=cv2.INTER_LINEAR)
dataset = temp

# 테스트 데이터 선택
sample = np.random.choice(len(dataset), 5, replace=False)
test = dataset[sample]

# Gaussian and salt&pepper noise
# 학습한 모델에 맞게 reshape
gaus = Gaussian(test)
gaus_test = gaus.reshape((len(test), 52, 40, 1))
salt = salt_and_pepper(test, 0.1)
salt_test = salt.reshape((len(salt), 52, 40, 1))

# gaussian or salt&pepper
noised = gaus
noised_test = gaus_test
# noised = salt
# noised_test = salt_test

model = keras.models.load_model('model_hybrid')  # Model 로드
dst = model.predict(noised_test)  # Input을 noised_test로 하여 예측
dst = dst.reshape(test.shape)  # Image 형식에 맞게 reshape

n = len(dst)
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n)
    plt.title(f'PSNR={psnr(test[i], salt[i]):.2f}')
    plt.imshow(noised[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.title(f'PSNR={psnr(test[i], dst[i].reshape(dataset.shape[1:])):.2f}')
    plt.imshow(dst[i].reshape(dataset.shape[1:]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()