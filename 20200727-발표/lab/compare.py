import numpy as np
import cv2
from scipy import ndimage
import sys


def nothing(x):
    pass


def gammaTrans(gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table


eps = sys.float_info.epsilon
def sigmoid_table(m=0.5, w=0.5, E=8):
    r = np.arange(0, 256) / 255.0
    s = (w / (1 + (m/(r+eps))**E)) + (1-w) * r
    return (255*s).astype(np.uint8)


def equalize(x):
    b, g, r = cv2.split(x)
    b_equal = cv2.equalizeHist(b)
    g_equal = cv2.equalizeHist(g)
    r_equal = cv2.equalizeHist(r)
    return cv2.merge([b_equal, g_equal, r_equal])


img = cv2.imread('data/dark.jpg')

gamma = 100; w = 0; E = 0; sigma = 1
cv2.namedWindow('img')
cv2.createTrackbar('gamma', 'img', gamma, 300, nothing)
cv2.createTrackbar('sigmoid:w', 'img', w, 100, nothing)
cv2.createTrackbar('sigmoid:E', 'img', E, 20, nothing)
cv2.createTrackbar('Equalization', 'img', 0, 1, nothing)
cv2.createTrackbar('Unsharp:sigma', 'img', sigma, 15, nothing)

cv2.namedWindow('sobel')
cv2.createTrackbar('abs', 'sobel', 0, 1, nothing)
th_min = 100; th_max = 200; size = 3
cv2.namedWindow('canny')
cv2.createTrackbar('th_min', 'canny', th_min, 512, nothing)
cv2.createTrackbar('th_max', 'canny', th_max, 512, nothing)
cv2.createTrackbar('size', 'canny', size, 9, nothing)
cv2.namedWindow('log')
cv2.createTrackbar('sigma', 'log', 1, 7, nothing)

dst = img
while(1):
    cv2.imshow('img', dst)
    k = cv2.waitKey(1)
    if k != -1:
        break

    gamma = cv2.getTrackbarPos('gamma', 'img') / 100
    w = cv2.getTrackbarPos('sigmoid:w', 'img') / 100
    E = cv2.getTrackbarPos('sigmoid:E', 'img')
    sigma = cv2.getTrackbarPos('Unsharp:sigma', 'img')
    if sigma < 1:
        cv2.setTrackbarPos('Unsharp:sigma', 'img', 1)
        sigma = 1
    g_table = gammaTrans(gamma);
    s_table = sigmoid_table(w=w, E=E);
    dst = cv2.LUT(img, g_table)
    dst = cv2.LUT(dst, s_table)
    if len(dst.shape) != 3 and cv2.getTrackbarPos('Equalization', 'img') != 0:
        cv2.setTrackbarPos('Equalization', 'img', 0)
    if cv2.getTrackbarPos('Equalization', 'img'):
        dst = equalize(dst)
    blur = cv2.GaussianBlur(dst / 255, ksize=(0, 0), sigmaX=sigma)
    dst = dst / 255 + (dst / 255 - blur)

    sobelx = cv2.Sobel(dst, cv2.CV_64F, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(dst, cv2.CV_64F, dx=0, dy=1, ksize=3)
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

    if cv2.getTrackbarPos('abs', 'sobel'):
        sobel = np.abs(sobelx) + np.abs(sobely)
    else:
        sobel = (sobelx_pm + sobely_pm) / 2

    log_sigma = cv2.getTrackbarPos('sigma', 'log')
    log = ndimage.gaussian_laplace(dst, sigma=sigma, mode='nearest')

    th_min = cv2.getTrackbarPos('th_min', 'canny')
    th_max = cv2.getTrackbarPos('th_max', 'canny')
    size = cv2.getTrackbarPos('size', 'canny')
    if size < 3:
        size = 3
        cv2.setTrackbarPos('size', 'canny', 3)
    elif size % 2 == 0:
        size += 1
        cv2.setTrackbarPos('size', 'canny', size)
    canny = cv2.Canny(np.uint8(dst*255), th_min, th_max, size)

    cv2.imshow('sobel', sobel)
    cv2.imshow('log', log)
    cv2.imshow('canny', canny)
