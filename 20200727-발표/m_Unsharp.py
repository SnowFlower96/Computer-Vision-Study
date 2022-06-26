import numpy as np
import cv2


def nothing(x):
    pass


img = cv2.imread('data/lenna.tif') / 255
mach = np.zeros(img.shape)
r = np.arange(0, mach.shape[1], int(mach.shape[1] / 5))
r[-1] = mach.shape[1]
col = np.arange(0, 1, 0.2)
for i in range(len(r)-1):
    mach[:,r[i]:r[i+1]+1] = col[i]
img = np.hstack((img, mach))

sigma = 3; scale = 1
blur = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=sigma)
unsharp = img - blur
sharped = img + scale * unsharp

cv2.namedWindow('UnsharpMasking')
cv2.createTrackbar('sigma', 'UnsharpMasking', 1, 15, nothing)
cv2.createTrackbar('scale', 'UnsharpMasking', 1, 5, nothing)

while(1):
    cv2.imshow('UnsharpMasking', sharped)
    k = cv2.waitKey(1)
    if k != -1:
        break

    sigma = cv2.getTrackbarPos('sigma', 'UnsharpMasking')
    scale = cv2.getTrackbarPos('scale', 'UnsharpMasking')
    if sigma <= 0:
        sigma = 1
        cv2.setTrackbarPos('sigma', 'UnsharpMasking', 1)
    if scale <= 0:
        scale = 1
        cv2.setTrackbarPos('scale', 'UnsharpMasking', 1)
    blur = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=sigma)
    unsharp = img - blur
    sharped = img + scale * unsharp
