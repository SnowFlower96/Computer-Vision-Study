import numpy as np
import cv2

img = cv2.imread('data/lenna.tif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

SobelHorizontal = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
SobelVertical = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

# cv2.Sobel
#SobelHorizontal = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#SobelVertical = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

kerH = np.array(SobelHorizontal, np.float)
kerV = np.array(SobelVertical, np.float)

dstH = cv2.filter2D(gray, -1, kerH)
dstV = cv2.filter2D(gray, -1, kerV)

plusH = np.where(dstH >= 128, dstH, 0); minusH = np.where(dstH < 128, dstH, 0)
plusV = np.where(dstV >= 128, dstV, 0); minusV = np.where(dstV < 128, dstV, 0)

sobelH_pm = np.zeros_like(img, dtype=float)
sobelH_pm[:,:,2] = plusH
sobelH_pm[:,:,0] = minusH
sobelV_pm = np.zeros_like(img, dtype=float)
sobelV_pm[:,:,2] = plusV
sobelV_pm[:,:,0] = minusV

cv2.imshow('H', sobelH_pm)
cv2.imshow('V', sobelV_pm)
cv2.waitKey()
