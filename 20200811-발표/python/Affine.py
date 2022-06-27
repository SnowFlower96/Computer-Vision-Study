import numpy as np
import cv2


def transfer(x, y):
    return np.array([[1, 0, x], [0, 1, y]], dtype=np.float)


def rotation(angle):
    theta = np.deg2rad(angle)
    return np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0]], dtype=np.float)


def rot(point, angle):
    if len(point) != 2:
        print("data wrong")
        exit(1)
    t = np.deg2rad(angle)
    c = np.cos(t); s = np.sin(t)
    return np.array([[c, s, (1-c)*point[0] - s*point[1]], [-s, c, s*point[0] + (1-c)*point[1]]])


def scale(s):
    return np.array([[1 * s, 0, 0], [0, 1 * s, 0]], dtype=np.float)


def shear(lx, ly):
    return np.array([[1, lx, 0], [ly, 1, 0]], dtype=np.float)


img = cv2.imread('lenna.jpg')
img = cv2.resize(img, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

dst = cv2.warpAffine(img, shear(0.2,0.2), dsize=(400,400))

cv2.imshow('dst', dst)
cv2.waitKey()