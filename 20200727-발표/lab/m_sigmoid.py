import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys

eps = sys.float_info.epsilon


def sigmoid_table(m=0.5, w=0.5, E=8):
    r = np.arange(0, 256) / 255.0
    s = (w / (1 + (m/(r+eps))**E)) + (1-w) * r
    return (255*r).astype(np.uint8), (255*s).astype(np.uint8)


def nothing(x):
    pass


img = cv2.imread('data/lenna.tif')


cv2.namedWindow('sigmoid')
cv2.createTrackbar('m', 'sigmoid', 50, 100, nothing)
cv2.createTrackbar('w', 'sigmoid', 0, 100, nothing)
cv2.createTrackbar('e', 'sigmoid', 0, 20, nothing)

m = cv2.getTrackbarPos('m', 'sigmoid')
w = cv2.getTrackbarPos('w', 'sigmoid')
e = cv2.getTrackbarPos('e', 'sigmoid')

_, table = sigmoid_table(m=m, w=w, E=e)
cv2.imshow('sigmoid', cv2.LUT(img, table))

while(True):
    cv2.imshow('sigmoid', cv2.LUT(img, table))
    k = cv2.waitKey(1)
    if k != -1:
        break

    m = cv2.getTrackbarPos('m', 'sigmoid') / 100
    w = cv2.getTrackbarPos('w', 'sigmoid') / 100
    e = cv2.getTrackbarPos('e', 'sigmoid')

    _, table = sigmoid_table(m=m, w=w, E=e)
    cv2.imshow('sigmoid', cv2.LUT(img, table))