import cv2
import numpy as np
import time


def gammaTrans(gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table


img = cv2.imread('data/fhd.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
stat = np.zeros([4,10])
gamma = [0.1, 0.3, 0.5, 0.7, 0.9, 2.0, 3.0, 4.0, 5.0, 6.0]
for i, g in enumerate(gamma):
    table = gammaTrans(g);
    st = time.time()
    a = (rgb / 255) ** g
    stat[0,i] = time.time() - st

    st = time.time()
    b = cv2.LUT(rgb,table)
    stat[1, i] = time.time() - st

    st = time.time()
    c = table[rgb]
    stat[2, i] = time.time() - st

    d = np.zeros_like(rgb)
    for m in range(rgb.shape[0]):
        for n in range(rgb.shape[1]):
            for c in range(rgb.shape[2]):
                d[m,n,c] = table[rgb[m,n,c]]
    stat[3, i] = time.time() - st

print(f'a : {np.mean(stat[0])}')
print(f'b : {np.mean(stat[1])}')
print(f'c : {np.mean(stat[2])}')
print(f'd : {np.mean(stat[3])}')