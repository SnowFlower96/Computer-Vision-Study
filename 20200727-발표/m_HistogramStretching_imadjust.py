import numpy as np
import cv2
from matplotlib import pyplot as plt

def imadjust(src, input_range=None, output_range=None, gamma=1):
    if input_range is None:
        input_range = (0, 1)
    low_in, high_in = input_range

    if output_range is None:
        output_range = (0, 1)
    low_out, high_out = output_range

    dst = np.empty_like(src, np.float)
    r = np.arange(0, 256, 1) / 255
    lut = np.zeros_like(r, np.float)
    for i in range(len(r)):
        if r[i] < low_in:
            lut[i] = low_out
        elif r[i] > high_in:
            lut[i] = high_out
        elif gamma == 1:
            lut[i] = (high_out - low_out) / (high_in - low_in) * (r[i] - low_in) + low_out
        else:
            lut[i] = (high_out - low_out) * ((r[i] - low_in)/ (high_in - low_in)) ** gamma + low_out

    row, col = src.shape[0:2]
    if len(src.shape) == 3:
        channel = 3
    else:
        channel = 1

    for i in range(0, row):
        for j in range(0, col):
            for k in range(0, channel):
                if src[i, j, k] < low_in:
                    dst[i, j, k] = low_out
                elif src[i, j, k] > high_in:
                    dst[i, j, k] = high_out
                elif gamma == 1:
                    dst[i, j, k] = (high_out - low_out) / (high_in - low_in) * (src[i, j, k] - low_in) + low_out
                else:
                    dst[i, j, k] = (high_out - low_out) * ((src[i, j, k] - low_in)/ (high_in - low_in)) ** gamma + low_out
    return dst, lut


img = cv2.imread('data/lenna.tif')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
copy = img.copy()
img = img/255
dst, lut = imadjust(img, input_range=(0.2, 0.7), output_range=(0, 1), gamma=1.5)

plt.subplot(221); plt.plot(np.arange(0, 256, 1) / 255, lut)
plt.subplot(222); plt.imshow(img); plt.axis('off'); plt.title('Original')
plt.subplot(223); plt.imshow(dst); plt.axis('off'); plt.title('For Loop')
plt.subplot(224); plt.imshow(cv2.LUT(copy, lut)); plt.axis('off'); plt.title('LUT')
print(np.array_equal(dst, cv2.LUT(copy, lut)))
plt.show()
