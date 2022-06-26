import cv2
import numpy as np
from matplotlib import pyplot as plt

size = [5,15,31]
img = cv2.imread('../../data/lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.ion()
plt.figure('mission')
plt.subplot(221)
plt.imshow(img)
plt.title('Original')
plt.axis('off')
plt.waitforbuttonpress()

for i, s in enumerate(size):
    kernel = np.ones((s,s)) / (s*s)
    dst = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_DEFAULT)
    plt.subplot(222+i)
    plt.imshow(dst)
    plt.title(f'{s}x{s}')
    plt.axis('off')
    plt.waitforbuttonpress()

plt.show()