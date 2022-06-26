import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('data/lenna.tif')
b, g, r = cv2.split(img)

b_equal = cv2.equalizeHist(b)
g_equal = cv2.equalizeHist(g)
r_equal = cv2.equalizeHist(r)

equal = cv2.merge([b_equal, g_equal, r_equal])
result = np.hstack((img, equal))

cv2.imshow('HistogramEqualization', result)
cv2.waitKey()


"""
hists = []
hists.append(np.histogram(b, 256, [0,255])[0])
hists.append(np.histogram(g, 256, [0,255])[0])
hists.append(np.histogram(r, 256, [0,255])[0])

cdfs = []; ncdfs = []; luts = []
for h in hists:
    c = h.cumsum()
    cdfs.append(c)
    ncdfs.append(c * h.max() / c.max())
    luts.append(c * 255 / c[255])

plt.subplot(531); plt.plot(np.arange(0,256,1), hists[0])
plt.subplot(532); plt.plot(np.arange(0,256,1), hists[1]); plt.title('Histogram')
plt.subplot(533); plt.plot(np.arange(0,256,1), hists[2])

plt.subplot(534); plt.plot(np.arange(0,256,1), cdfs[0])
plt.subplot(535); plt.plot(np.arange(0,256,1), cdfs[1]); plt.title('CDF')
plt.subplot(536); plt.plot(np.arange(0,256,1), cdfs[2])

plt.subplot(537); plt.plot(np.arange(0,256,1), ncdfs[0])
plt.subplot(538); plt.plot(np.arange(0,256,1), ncdfs[1]); plt.title('NCDF')
plt.subplot(539); plt.plot(np.arange(0,256,1), ncdfs[2])

plt.subplot(5,3,10); plt.plot(np.arange(0,256,1), luts[0])
plt.subplot(5,3,11); plt.plot(np.arange(0,256,1), luts[1]); plt.title('LUT')
plt.subplot(5,3,12); plt.plot(np.arange(0,256,1), luts[2])

plt.subplot(5,3,13); plt.imshow(cv2.LUT(b, luts[0]))
plt.subplot(5,3,14); plt.imshow(cv2.LUT(g, luts[1]))
plt.subplot(5,3,15); plt.imshow(cv2.LUT(r, luts[2]))
plt.show()
"""