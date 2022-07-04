import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import numpy as np
import matplotlib.pylab as plt
from time import time
import math


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    MAX = np.max(img1)
    return 20 * math.log10(MAX) - 10 * math.log10(mse)


img1 = rgb2gray(imread('images/1.png'))
h1, w1 = img1.shape
img2 = rgb2gray(imread('images/2.png'))
h2, w2 = img2.shape
img3 = rgb2gray(imread('images/3.jpg'))
h3, w3 = img3.shape
img4 = rgb2gray(imread('images/inter.jpg'))
h4, w4 = img4.shape
print(h4, w4), exit(0)
# 이미지에 노이즈 적용
print('Distorting image...')
distorted1 = img1.copy()
distorted2 = img2.copy()
distorted3 = img3.copy()
distorted4 = img4.copy()
distorted1 += 0.2 * np.random.randn(h1, w1)
distorted2 += 0.2 * np.random.randn(h2, w2)
distorted3 += 0.2 * np.random.randn(h3, w3)
distorted4 += 0.2 * np.random.randn(h4, w4)

print('Extracting reference patches...')
t0 = time()
patch_size = (7, 7)
data = extract_patches_2d(distorted1, patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
# data.shape = (22256, 49) type = ndarray
print('done in %.2fs.' % (time() - t0))

# 참조 패치에서 dictionary learning 진행, 결과물은 256개
print('Learning the dictionary...')
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=256, alpha=1, n_iter=600)
V = dico.fit(data).components_
dt = time() - t0
print('done in %.2fs.' % dt)

# 노이즈 이미지에서 패치를 추출
print('Extracting noisy patches... ')
t0 = time()
data1 = extract_patches_2d(distorted1, patch_size)
data1 = data1.reshape(data1.shape[0], -1)
intercept = np.mean(data1, axis=0)
data1 -= intercept

data2 = extract_patches_2d(distorted2, patch_size)
data2 = data2.reshape(data2.shape[0], -1)
intercept = np.mean(data2, axis=0)
data2 -= intercept

data3 = extract_patches_2d(distorted3, patch_size)
data3 = data3.reshape(data3.shape[0], -1)
intercept = np.mean(data3, axis=0)
data3 -= intercept

data4 = extract_patches_2d(distorted4, patch_size)
data4 = data4.reshape(data4.shape[0], -1)
intercept = np.mean(data4, axis=0)
data4 -= intercept
print('done in %.2fs.' % (time() - t0))

# 이미지 복원
print('Orthogonal Matching Pursuit\n2 atoms' + '...')
kwargs = {'transform_n_nonzero_coefs': 2}
reconstruction1 = img1.copy()
reconstruction2 = img2.copy()
reconstruction3 = img3.copy()
reconstruction4 = img4.copy()
t0 = time()
dico.set_params(transform_algorithm='omp', **kwargs)

code = dico.transform(data1)
patches1 = np.dot(code, V)
patches1 += intercept
patches1 = patches1.reshape(len(data1), *patch_size)
reconstruction1 = reconstruct_from_patches_2d(patches1, (h1, w1))

code = dico.transform(data2)
patches2 = np.dot(code, V)
patches2 += intercept
patches2 = patches2.reshape(len(data2), *patch_size)
reconstruction2 = reconstruct_from_patches_2d(patches2, (h2, w2))

code = dico.transform(data3)
patches3 = np.dot(code, V)
patches3 += intercept
patches3 = patches3.reshape(len(data3), *patch_size)
reconstruction3 = reconstruct_from_patches_2d(patches3, (h3, w3))

code = dico.transform(data4)
patches4 = np.dot(code, V)
patches4 += intercept
patches4 = patches4.reshape(len(data4), *patch_size)
reconstruction4 = reconstruct_from_patches_2d(patches4, (h4, w4))
dt = time() - t0
distorted1 = cv2.normalize(distorted1, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
distorted2 = cv2.normalize(distorted2, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
distorted3 = cv2.normalize(distorted3, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
distorted4 = cv2.normalize(distorted4, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

reconstruction1 = cv2.normalize(reconstruction1, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
reconstruction2 = cv2.normalize(reconstruction2, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
reconstruction3 = cv2.normalize(reconstruction3, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
reconstruction4 = cv2.normalize(reconstruction4, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
print('done in %.2fs.' % dt)
# 결과물 출력
plt.subplot(431); plt.imshow(img1, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'Original')
plt.subplot(432); plt.imshow(distorted1, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'{psnr(img1, distorted1):.2f}')
plt.subplot(433); plt.imshow(reconstruction1, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'{psnr(img1, reconstruction1):.2f}')

plt.subplot(434); plt.imshow(img2, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'Original')
plt.subplot(435); plt.imshow(distorted2, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'{psnr(img2, distorted2):.2f}')
plt.subplot(436); plt.imshow(reconstruction2, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'{psnr(img2, reconstruction2):.2f}')

plt.subplot(437); plt.imshow(img3, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'Original')
plt.subplot(438); plt.imshow(distorted3, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'{psnr(img3, distorted3):.2f}')
plt.subplot(439); plt.imshow(reconstruction3, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'{psnr(img3, reconstruction3):.2f}')

plt.subplot(4,3,10); plt.imshow(img4, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'Original')
plt.subplot(4,3,11); plt.imshow(distorted4, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'{psnr(img4, distorted4):.2f}')
plt.subplot(4,3,12); plt.imshow(reconstruction4, cmap=plt.cm.gray); plt.axis('off'); plt.title(f'{psnr(img4, reconstruction4):.2f}')
plt.show()