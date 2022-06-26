import numpy as np
import cv2

img = cv2.imread('data/lenna.tif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255
canvas = np.zeros_like(img, float)

lap = cv2.Laplacian(gray, -1, ksize=3, scale=3)
plus = np.where(lap > 0, lap, 0); minus = np.where(lap < 0, lap, 0)

canvas[:,:,2] = plus
canvas[:,:,0] = np.abs(minus)

cv2.imshow('Laplacian', lap)
cv2.imshow('canvas', canvas)
cv2.waitKey()
