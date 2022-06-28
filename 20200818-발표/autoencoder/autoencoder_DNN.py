import keras
import numpy as np
from sklearn.datasets import fetch_lfw_people


# Gaussian 노이즈 이미지 변환 함수
def noise(img):
    mean = 0
    var = 0.01
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape).reshape(img.shape)
    return img + gauss


num_epochs = 100
batch_size = 16
learning_rate = 1e-3

# fetch_lfw_people : 5794 classes, 13233 samples
dataset = fetch_lfw_people(min_faces_per_person=70, resize=0.4).images / 255
# print(dataset.shape) # (1288, 50, 37)

# 신경망 입력을 위한 flatten, (1288, 50, 37) -> (1288, 1850)
# flatten 후 가우시안 노이즈 적용
flat = dataset.reshape(len(dataset), np.prod(dataset.shape[1:]))
noised = noise(flat)

# input node의 수 : 이미지 한장의 픽셀수
input_img = keras.layers.Input(shape=(flat.shape[1],))  # 함수형 API의 경우 입력 node와 모양을 정의
encoded = keras.layers.Dense(512, activation='relu')(input_img)
middle1 = keras.layers.Dense(128, activation='relu')(encoded)
middle2 = keras.layers.Dense(512, activation='relu')(middle1)
decoded = keras.layers.Dense(flat.shape[1], activation='sigmoid')(middle2)

autoencoder = keras.models.Model(input_img, decoded)

optimizer = keras.optimizers.adam(lr=learning_rate, decay=1e-5)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')
autoencoder.fit(noised, flat, batch_size=batch_size, epochs=num_epochs, shuffle=True)

autoencoder.save("model1")
