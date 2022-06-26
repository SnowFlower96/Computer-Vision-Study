import cv2
import numpy as np
from matplotlib import pyplot as plt


def getGaussian(k, sigma):
    r = np.arange(-int(k / 2), k - int(k / 2), 1)
    g = np.exp(-r**2/(2*sigma**2))
    g = g * (1/np.sum(g))
    return r, g


img = cv2.imread('../../data/lenna.png')

r, g = getGaussian(11, 3)
kernel = cv2.getGaussianKernel(11, 3, cv2.CV_32F)
print(np.sum(g * (1/np.sum(g))))
g = g * (1/np.sum(g))
plt.figure(num='mission2')
plt.subplot(121)
plt.plot(r, g, 'b')
plt.subplot(122)
plt.plot(r, kernel, 'r')
plt.show()