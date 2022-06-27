import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass


def get_filter_mask(mask_size, d=20, sigma=5, high = 0):
    rows, cols = mask_size
    crow = rows // 2; ccol = cols // 2

    if high:
        mask = np.ones((rows, cols), dtype=float)
        for x in range(-d, d+1):
            for y in range(-d, d+1):
                if x**2 + y **2 < d**2:
                    mask[crow+y, ccol+x] = 0
    else:
        mask = np.zeros((rows, cols), dtype=float)
        for x in range(-d, d + 1):
            for y in range(-d, d + 1):
                if x ** 2 + y ** 2 < d ** 2:
                    mask[crow + y, ccol + x] = 1
    if sigma > 0:
        mask_blr = cv2.GaussianBlur(mask, (21, 21), 5)

    return mask_blr, mask


path = 'data/'
name = 'lenna.tif'
img = cv2.imread(path + name)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

d = 30
cv2.namedWindow('FT')
cv2.createTrackbar('d', 'FT', d, 200, nothing)
cv2.createTrackbar('high', 'FT', 0, 1, nothing)

while(1):
    k = cv2.waitKey(1)
    if k != -1 and k == ord('q'):
        break

    d = cv2.getTrackbarPos('d', 'FT')
    h = cv2.getTrackbarPos('high', 'FT')
    if d < 1:
        cv2.setTrackbarPos('d', 'FT', 1)
        d = 1

    f = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f)

    mask_blr, mask = get_filter_mask(gray.shape, d=d, high=h)
    fshift = f_shift * mask_blr
    spectrum = 20 * np.log(np.abs(fshift) + 1)

    f_ishift = np.fft.ifftshift(fshift)
    img_flt = np.fft.ifft2(f_ishift)
    img_flt = np.abs(img_flt)

    dst = img_flt / np.max(img_flt)
    spec = spectrum / np.max(spectrum)
    result = np.hstack((dst, spec))
    cv2.imshow('FT', result)

