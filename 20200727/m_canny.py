import cv2


def nothing(x):
    pass


img = cv2.imread('data/lenna.tif', 0)

th_min = 100
th_max = 200
size = 3
cv2.namedWindow('Canny')
cv2.createTrackbar('th_min', 'Canny', th_min, 512, nothing)
cv2.createTrackbar('th_max', 'Canny', th_max, 512, nothing)
cv2.createTrackbar('size', 'Canny', size, 9, nothing)
cv2.createTrackbar('norm', 'Canny', 0, 1, nothing)

edges = cv2.Canny(img, th_min, th_max, size, L2gradient=False)

while(True):
    cv2.imshow('Canny', edges)
    k = cv2.waitKey(1)  # & 0xFF
    if k != -1:  # break if any key in input
        break

    th_min = cv2.getTrackbarPos('th_min', 'Canny')
    th_max = cv2.getTrackbarPos('th_max', 'Canny')
    size = cv2.getTrackbarPos('size', 'Canny')
    L2 = cv2.getTrackbarPos('norm', 'Canny')

    if size < 3:
        size = 3
        cv2.setTrackbarPos('size', 'Canny', 3)
        print(size)
    elif size % 2 == 0:
        size += 1
        cv2.setTrackbarPos('size', 'Canny', size)

    edges = cv2.Canny(img, th_min, th_max, size, L2gradient=L2)
