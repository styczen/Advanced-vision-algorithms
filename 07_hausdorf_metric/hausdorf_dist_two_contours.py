import matplotlib.pyplot as plt
import os
import cv2
import sys
import numpy as np

DIR = os.path.dirname(sys.argv[0])


def dist(c):
    xy=c[0][:, 0, :]

    m = cv2.moments(img, True)
    x_center = m['m10'] / m['m00']
    y_center = m['m01'] / m['m00']

    xy_centered = xy - np.array([x_center, y_center])

    max_dist = 0.0
    for row in xy_centered:
        for row1 in xy_centered:
            dist = (row[0] - row1[0]) ** 2 + (row[1] - row1[1]) ** 2
            if dist > max_dist:
                max_dist = dist

    return xy_centered, x_center, y_center


#####
img = cv2.imread(DIR + '/plikiHausdorff/imgs/c_astipalea.bmp')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = 255 - img
gray = 255 - gray

img1, contours, hierarchy = cv2.findContours(image=img,
                                             mode=cv2.RETR_TREE,
                                             method=cv2.CHAIN_APPROX_NONE)

img_contours = cv2.drawContours(image=img,
                                 contours=contours,
                                 contourIdx=0,
                                 color=(255, 0, 0))

xy_centered, x_center, y_center = dist(contours)

gray = gray /

# plt.figure()
# plt.imshow(img_contours)
#
# plt.figure()
# plt.imshow(gray, 'gray')
#
# plt.show()


# cv2.imshow('kek', img_contours)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()