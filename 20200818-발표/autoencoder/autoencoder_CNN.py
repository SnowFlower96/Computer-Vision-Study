import keras
import numpy as np
from sklearn.datasets import fetch_lfw_people
import cv2


def noise(img):
    mean = 0
    var = 0.1
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape).reshape(img.shape)
    return img + gauss


num_epochs = 100
batch_size = 16
learning_rate = 1e-3

# fetch_lfw_people : 5794 classes, 13233 samples
dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4).images / 255
# print(dataset.shape) # (1288, 50, 37)

# maxpooling과 upsampling을 (2, 2)로 진행하므로 2의 n승으로 나누어지게 resize
temp = np.zeros((len(dataset), 52, 40))
for i, d in enumerate(dataset):
    temp[i] = cv2.resize(d, (40, 52), interpolation=cv2.INTER_LINEAR)
dataset = temp

# CNN입력을 위한 reshape, 영상의 원본이 유지되지만, 1차원이 추가됨
dataset = dataset.reshape((len(dataset), dataset.shape[1], dataset.shape[2], 1))
noised = noise(dataset)

# input 계층 정의
input_img = keras.layers.Input(shape=(dataset.shape[1], dataset.shape[2], 1))

# encoding 단계에서는 convolution과 maxplooling을 2회씩 실시
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

# decoding 단계에서는 convolution과 upsampling을 2회씩 실시
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = keras.layers.UpSampling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = keras.layers.UpSampling2D((2, 2))(x)
decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

# autoencoder 모델 정의, input과 output을 정해주면 자동으로 내부 신경망 생성
autoencoder = keras.models.Model(input_img, decoded)
# adam optimizer의 계수 조절
optimizer = keras.optimizers.adam(lr=learning_rate, decay=1e-5)
# 최적화 함수와 손실 함수가 정의된 모델 생성
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
# input data는 noised, 확인 데이터는 dataset 설정
autoencoder.fit(noised, dataset, batch_size=batch_size, epochs=num_epochs, shuffle=True)
# 최종 학습된 모델 저장
autoencoder.save("model_conv2d")

