import cv2
import numpy as np


# 이미지의 중심을 기준으로 90도 회전
def rotation(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((h/2, w/2), 90, 1)
    output = cv2.warpAffine(img, M, (h, w))
    return output


def affine(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    pts2 = np.float32([[0, 0], [w, h / 15], [w / 15, h]])
    M = cv2.getAffineTransform(pts1, pts2)
    output = cv2.warpAffine(img, M, (w, h))
    return output


# def perspective(img):
#     h, w = img.shape[:2]
#     pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
#     pts2 = np.float32([[0, 0], [w / 15, h], [w, h / 15], [w / 10, h / 10]])
#     M = cv2.getPerspectiveTransform(pts1, pts2)
#     output = cv2.warpAffine(img, M, (h, w))
#     return output
