from skimage.io import imread
from skimage.color import rgb2gray
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import numpy as np
import matplotlib.pylab as plt
from time import time


# image와 reference의 차이를 출력
def show_with_diff(image, reference, title):
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.title('Image')
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')
    plt.subplot(122)
    difference = image - reference
    plt.title('Difference (norm: %.2f)' % np.sqrt(np.sum(difference ** 2)))
    plt.imshow(difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')
    plt.suptitle(title, size=20)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)
    plt.show()


img = rgb2gray(imread('lena.png'))
height, width = img.shape

# 이미지 아래쪽에 노이즈 발생
print('Distorting image...')
distorted = img.copy()
distorted[height // 2:, :] += 0.085 * np.random.randn(height // 2, width)

# 이미지 아래쪽에서 참조패치 추출
print('Extracting reference patches...')
t0 = time()
patch_size = (7, 7)
data = extract_patches_2d(distorted[height // 2:, :], patch_size)
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
# V.shape = (256, 49) type = ndarray
dt = time() - t0
print('done in %.2fs.' % dt)

# Dictionary learning의 결과물 출력
plt.figure(figsize=(5, 5))
for i, comp in enumerate(V):
    plt.subplot(16, 16, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')
plt.suptitle('Dictionary learned from lena patches\n' +
             'Train time %.1fs on %d patches' % (dt, len(data)),
             fontsize=20)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

# 원본 이미지와 노이즈된 이미지 출력
show_with_diff(distorted, img, 'Distorted image')

# 노이즈 된 이미지 아래부분에서 패치를 추출
print('Extracting noisy patches... ')
t0 = time()
data = extract_patches_2d(distorted[height // 2:, :], patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
print('done in %.2fs.' % (time() - t0))

# Orthognal Matching Pursuit(일치추적법)을 사용하여 이미지 복원
print('Orthogonal Matching Pursuit\n2 atoms' + '...')
kwargs = {'transform_n_nonzero_coefs': 2}
reconstruction = img.copy()
t0 = time()
dico.set_params(transform_algorithm='omp', **kwargs)
code = dico.transform(data)
# code.shape = (22256, 256), type = ndarray
patches = np.dot(code, V)
patches += intercept
patches = patches.reshape(len(data), *patch_size)
reconstruction[height // 2:, :] = reconstruct_from_patches_2d(patches, (height // 2, width))
dt = time() - t0

print('done in %.2fs.' % dt)
# 결과물 출력
show_with_diff(reconstruction, img, 'Orthogonal Matching Pursuit\n2 atoms (time: %.1fs)' % dt)