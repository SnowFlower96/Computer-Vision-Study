import numpy as np
import cv2


def nothing(x):
    pass


img = cv2.imread('data/Hue_color_wheel_by_degree.png')
# img = cv2.imread('data/fruits.jpg')
hsv = cv2.cvtColor(img.astype('float32') / 255, cv2.COLOR_BGR2HSV)
result = hsv.copy()

is_r = False; is_g = False; is_b = False
r = [0, 60, 300, 360]; g = [60, 180]; b = [180, 300]
h_val = 0; s_val = 0; v_val = 0
cv2.namedWindow('HSV')
cv2.createTrackbar('H', 'HSV', h_val, 360, nothing)
cv2.createTrackbar('S', 'HSV', s_val, 100, nothing)
cv2.createTrackbar('V', 'HSV', v_val, 100, nothing)

while(1):
    h, s, v = cv2.split(result)
    cv2.imshow('HSV', cv2.cvtColor(result, cv2.COLOR_HSV2BGR))
    k = cv2.waitKey(1)
    if k == 27 or k == ord('q'):
        break
    if k == ord('r'):
        if is_r is True:
            is_r = False
        else:
            is_r = True
        print(f'R is {is_r}')
    if k == ord('g'):
        if is_g is True:
            is_g = False
        else:
            is_g = True
        print(f'G is {is_g}')
    if k == ord('b'):
        if is_b is True:
            is_b = False
        else:
            is_b = True
        print(f'B is {is_b}')

    h_val = cv2.getTrackbarPos('H', 'HSV')
    s_val = cv2.getTrackbarPos('S', 'HSV') / 100
    v_val = cv2.getTrackbarPos('V', 'HSV') / 100

    h = hsv[:, :, 0] + h_val
    h = np.where(h > 360, h - 360, h)
    s = hsv[:, :, 1] + s_val
    s = np.where(s > 360, s - 360, s)
    v = hsv[:, :, 2] + v_val
    v = np.where(v > 360, v - 360, v)
    result = cv2.merge([h, s, v])

    if is_r or is_g or is_b:
        mask_r = cv2.inRange(h, r[0], r[1]) | cv2.inRange(h, r[2], r[3])
        mask_g = cv2.inRange(h, g[0], g[1])
        mask_b = cv2.inRange(h, b[0], b[1])
        mask = np.zeros_like(mask_r)

        if is_r:
            mask = mask | mask_r
        if is_g:
            mask = mask | mask_g
        if is_b:
            mask = mask | mask_b
        result = cv2.bitwise_and(result, result, mask=mask)
