import cv2
import numpy as np
import sys
import os

DIR = os.path.dirname(sys.argv[0])


def nothing(x):
    pass


img = cv2.imread(DIR + '/07_hausdorf_metric/plikiHausdorff/Aegeansea.jpg')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600, 600)

# create trackbars for color change
cv2.createTrackbar('H_low', 'image', 0, 255, nothing)
cv2.createTrackbar('H_high', 'image', 0, 255, nothing)

cv2.createTrackbar('S_low', 'image', 0, 255, nothing)
cv2.createTrackbar('S_high', 'image', 0, 255, nothing)

cv2.createTrackbar('V_low', 'image', 0, 255, nothing)
cv2.createTrackbar('V_high', 'image', 0, 255, nothing)

cv2.setTrackbarPos('H_high', 'image', 255)
cv2.setTrackbarPos('S_high', 'image', 255)
cv2.setTrackbarPos('V_high', 'image', 255)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
while True:
    # get current positions of four trackbars
    h_low = cv2.getTrackbarPos('H_low', 'image')
    s_low = cv2.getTrackbarPos('S_low', 'image')
    v_low = cv2.getTrackbarPos('V_low', 'image')
    h_high = cv2.getTrackbarPos('H_high', 'image')
    s_high = cv2.getTrackbarPos('S_high', 'image')
    v_high = cv2.getTrackbarPos('V_high', 'image')

    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('image', mask)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import sys
# import os
#
# DIR = os.path.dirname(sys.argv[0])
#
# # from matplotlib import pyplot as plt
#
#
# def nothing(x):
#     pass
#
#
# cv2.namedWindow('Colorbars')
#
# hh = 'Max'
# hl = 'Min'
# wnd = 'Colorbars'
#
# cv2.createTrackbar("Max", "Colorbars", 0, 255, nothing)
# cv2.createTrackbar("Min", "Colorbars", 0, 255, nothing)
#
# img = cv2.imread(DIR + '/07_hausdorf_metric/plikiHausdorff/Aegeansea.jpg')
# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
#
# # titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# # images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#
# # for i in xrange(6):
# #     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
# #     plt.title(titles[i])
# #     plt.xticks([]),plt.yticks([])
#
# # plt.show()
#
# while True:
#     hul = cv2.getTrackbarPos("Max", "Colorbars")
#     huh = cv2.getTrackbarPos("Min", "Colorbars")
#     h_low = cv2.getTrackbarPos('H_low', 'image')
#     s_low = cv2.getTrackbarPos('S_low', 'image')
#     v_low = cv2.getTrackbarPos('V_low', 'image')
#
#     ret, thresh1 = cv2.threshold(img, hul, huh, cv2.THRESH_BINARY)
#     ret, thresh2 = cv2.threshold(img, hul, huh, cv2.THRESH_BINARY_INV)
#     ret, thresh3 = cv2.threshold(img, hul, huh, cv2.THRESH_TRUNC)
#     ret, thresh4 = cv2.threshold(img, hul, huh, cv2.THRESH_TOZERO)
#     ret, thresh5 = cv2.threshold(img, hul, huh, cv2.THRESH_TOZERO_INV)
#     # cv2.imshow(wnd)
#     cv2.imshow("thresh1", thresh1)
#     cv2.imshow("thresh2", thresh2)
#     cv2.imshow("thresh3", thresh3)
#     cv2.imshow("thresh4", thresh4)
#     cv2.imshow("thresh5", thresh5)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('m'):
#         mode = not mode
#     elif k == 27:
#         break
#
# cv2.destroyAllWindows()