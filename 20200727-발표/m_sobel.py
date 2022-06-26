import numpy as np
import cv2

img = cv2.imread('data/lenna.tif')
img = img/255

sobelx = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)

scale = 1
minx = scale * np.min(sobelx); maxx = np.max(sobelx)
plusx = scale * np.clip(sobelx, 0, maxx) / maxx
minusx = np.clip(sobelx, minx, 0) / abs(minx)
minusx = np.clip(minusx, -1, 0)
sobelx_pm = (plusx + (minusx + 1)) / 2

miny = np.min(sobely); maxy = np.max(sobely)
plusy = scale * np.clip(sobely, 0, maxy) / maxy
minusy = scale * np.clip(sobely, miny, 0) / abs(miny)
minusy = np.clip(minusy, -1, 0)
sobely_pm = (plusy + (minusy + 1)) / 2

# cv2.imshow('original', img)
# cv2.imshow('soblex_pm', sobelx_pm)
# cv2.imshow('sobely_pm', sobely_pm)
cv2.imshow('abs', (np.abs(sobelx) + np.abs(sobely)))
cv2.imshow('sobel', (sobelx_pm + sobely_pm)/2)
cv2.waitKey()

