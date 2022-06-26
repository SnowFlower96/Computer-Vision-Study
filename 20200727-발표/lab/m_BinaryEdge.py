import numpy as np
import cv2


def nothing(x):
    pass


img = cv2.imread('data/lenna.tif')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = gray / 255

cv2.imshow('image', gray)
cv2.createTrackbar('Threshold', 'image', 30, 1024, nothing)
cv2.createTrackbar('+ edge', 'image', 1, 1, nothing)
cv2.createTrackbar('- edge', 'image', 1, 1, nothing)

SobelHorizontal = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
SobelVertical = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# cv2.Sobel()
#SobelHorizontal = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#SobelVertical = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

edgeH = cv2.filter2D(gray, -1, SobelHorizontal)
edgeV = cv2.filter2D(gray, -1, SobelVertical)

while(1):
    imgEdge = np.zeros_like(gray, 'bool')
    k = cv2.waitKey(1)
    if k != -1:
        break
    Threshold = cv2.getTrackbarPos('Threshold', 'image') / 255

    edgeHplus = edgeH > Threshold  # 가로 에지 + 성분. bool image.
    edgeHminus = edgeH < -Threshold  # 가로 에지 - 성분. bool image.
    edgeVplus = edgeV > Threshold  # 세로 에지 + 성분. bool image.
    edgeVminus = edgeV < -Threshold  # 세로 에지 - 성분. bool image.

    edgePlus = edgeHplus + edgeVplus
    edgeMinus = edgeHminus + edgeVminus
    # tmp = 1.0 * edgePlus; cv2.imshow('edgePlus', tmp)

    if cv2.getTrackbarPos('+ edge', 'image') == 1:
        imgEdge = imgEdge | edgePlus
    if cv2.getTrackbarPos('- edge', 'image') == 1:
        imgEdge = imgEdge | edgeMinus

    # bool 영상은 출력이 안되므로 부동소수 영상으로 만들어 출력한다. 편의상 역상으로 변환하여 출력한다.
    cv2.imshow('image', 1 - 1.0 * imgEdge)
    # cv2.imshow('image', imgEdge.astype(np.float))