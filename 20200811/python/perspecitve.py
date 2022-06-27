import numpy as np
import cv2
import time
import math


def mouse(event, x, y, flags, param):
    global hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y}) : {hsv[y,x,:]}')


def centeroidpython(data):
    x, y = zip(*data)
    l = len(x)
    return sum(x) / l, sum(y) / l


img = cv2.imread('lenna.jpg')
video1 = cv2.VideoCapture('video.avi')
fps1, cnt1 = video1.get(cv2.CAP_PROP_FPS), video1.get(cv2.CAP_PROP_FRAME_COUNT)
width1, height1 = video1.get(cv2.CAP_PROP_FRAME_WIDTH), video1.get(cv2.CAP_PROP_FRAME_HEIGHT)

video2 = cv2.VideoCapture('car_race1.mp4')
fps2, cnt2 = video2.get(cv2.CAP_PROP_FPS), video2.get(cv2.CAP_PROP_FRAME_COUNT)
width1, height2 = video2.get(cv2.CAP_PROP_FRAME_WIDTH), video2.get(cv2.CAP_PROP_FRAME_HEIGHT)

cv2.namedWindow('video')
cv2.setMouseCallback('video', mouse)

low = np.array([100, 100, 0])
high = np.array([120, 190, 255])
while True:
    st = time.time()
    _, frame1 = video1.read()
    _, frame2 = video2.read()
    result = frame1.copy()

    if video1.get(cv2.CAP_PROP_POS_FRAMES) == cnt1:
        break
    if video2.get(cv2.CAP_PROP_POS_FRAMES) == cnt2:
        video2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.dilate(mask, None, iterations=2)
    mask_inv = 255 - mask
    masked_hsv = cv2.bitwise_and(hsv, hsv, mask=mask)
    masked_bgr = cv2.cvtColor(masked_hsv, cv2.COLOR_HSV2BGR)
    masked_gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)

    _, imgBin = cv2.threshold(masked_gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(frame1, contours, -1, (255, 0, 0), 3)

    for c in contours:
        a = cv2.approxPolyDP(c, 100, True)
        if len(a) == 4:
            cv2.drawContours(result, [a], -1, (0, 0, 255), 5)
            pts = a.squeeze()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]

            centroid_x, centroid_y = centeroidpython(pts)
            s_pts1 = np.array(sorted(pts, key=lambda x: math.atan2((x[1] - centroid_y), (x[0] - centroid_x))), dtype=np.float32)

            dist_x = int(math.dist(s_pts1[0], s_pts1[1]))
            dist_y = int(math.dist(s_pts1[0], s_pts1[3]))

            for p, color in zip(s_pts1, colors):
                cv2.circle(result, tuple(p.astype(int)), 10, color=color, thickness=-1)

            pic = frame2.copy()
            pic = cv2.resize(pic, dsize=(dist_x, dist_y), interpolation=cv2.INTER_LINEAR)
            h, w = pic.shape[:2]
            pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            pts2[:, 0] += s_pts1[0, 0]
            pts2[:, 1] += s_pts1[0, 1]

            t_matrix = np.array([[1, 0, s_pts1[0, 0]], [0, 1, s_pts1[0, 1]]], dtype=np.float)
            pic_trans = cv2.warpAffine(pic, t_matrix, dsize=(result.shape[1], result.shape[0]))
            p_matrix = cv2.getPerspectiveTransform(pts2, s_pts1)
            pic_pers = cv2.warpPerspective(pic_trans, p_matrix, dsize=(result.shape[1], result.shape[0]))
            break

    pic_pers = cv2.bitwise_and(pic_pers, pic_pers, mask=mask)

    result = cv2.bitwise_and(result, result, mask=mask_inv)
    result += pic_pers

    cv2.imshow('video', result)

    k = cv2.waitKey(1)
    if k != -1 and k == ord('q'):
        break

video1.release()
cv2.destroyAllWindows()

