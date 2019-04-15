import cv2
import numpy as np
import os
import sys

DIR = os.path.dirname(sys.argv[0])

trybik = cv2.imread(DIR + '/obrazy_hough/trybik.jpg')
trybik_gray =cv2.cvtColor(trybik, cv2.COLOR_BGR2GRAY)
trybik_gray = 255 - trybik_gray

_, trybik_mask = cv2.threshold(trybik_gray, 100, 255, cv2.THRESH_BINARY)
trybik_mask = cv2.medianBlur(trybik_mask, 5)

m = cv2.moments(trybik_mask, True)
x_c = m['m10'] / m['m00']
y_c = m['m01'] / m['m00']

# REMEMBER TO ADD THIRD RETURN PARAM IN FRONT BECAUSE OF DIFFERENT OPENCV VERSIONS
_, contours, hierarchy = cv2.findContours(image=trybik_mask,
                                          mode=cv2.RETR_TREE,  # first contour should be the longest
                                          method=cv2.CHAIN_APPROX_NONE)

trybik = cv2.drawContours(image=trybik,
                          contours=contours,
                          contourIdx=0,
                          color=(255, 0, 0))

cv2.circle(img=trybik,
           center=(np.int(x_c), np.int(y_c)),
           radius=2,
           color=(0, 255, 0),
           thickness=2)

sobelx = cv2.Sobel(trybik_gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(trybik_gray, cv2.CV_64F, 0, 1, ksize=5)

gradient = np.sqrt(sobelx ** 2 + sobely ** 2)
gradient = gradient / np.max(gradient)

orientation = np.rad2deg(np.arctan2(sobely, sobelx))
orientation += 180
orientation = np.uint16(orientation)

Rtable = [[] for i in range(360)]

for point in contours[0]:
    omega = orientation[point[0, 0], point[0, 1]]
    r = np.sqrt((point[0, 0] - x_c) ** 2 + (point[0, 1] - y_c) ** 2)
    beta = np.arctan2(point[0, 1] - y_c, point[0, 0] - x_c)
    Rtable[omega-1].append((r, beta))


####
trybik2 = cv2.imread(DIR + '/obrazy_hough/trybiki2.jpg')
trybik2_gray =cv2.cvtColor(trybik2, cv2.COLOR_BGR2GRAY)

sobelx2 = cv2.Sobel(trybik2_gray, cv2.CV_64F, 1, 0, ksize=5)
sobely2 = cv2.Sobel(trybik2_gray, cv2.CV_64F, 0, 1, ksize=5)

gradient2 = np.sqrt(sobelx2 ** 2 + sobely2 ** 2)
gradient2 = gradient2 / np.max(gradient2)




cv2.namedWindow('trybik', cv2.WINDOW_NORMAL)
cv2.resizeWindow('trybik', 300, 300)
cv2.imshow('trybik', trybik)

cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)
cv2.resizeWindow('gradient', 300, 300)
cv2.imshow('gradient', gradient)

cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask', 300, 300)
cv2.imshow('mask', trybik_mask)

cv2.waitKey(0)

cv2.destroyAllWindows()
