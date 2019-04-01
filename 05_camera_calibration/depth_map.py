import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

DIR = os.getcwd() + '/calibration_stereo'

aloe_l = cv2.imread(DIR + '/aloes/aloeL.jpg')
aloe_r = cv2.imread(DIR + '/aloes/aloeR.jpg')

aloe_l_gray = cv2.cvtColor(aloe_l, cv2.COLOR_BGR2GRAY)
aloe_r_gray = cv2.cvtColor(aloe_r, cv2.COLOR_BGR2GRAY)

# cv2.namedWindow('l', cv2.WINDOW_GUI_NORMAL)
# cv2.imshow('l', aloe_l_gray)
# cv2.imshow('r', aloe_r_gray)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

stereoBM = cv2.StereoBM_create(numDisparities=128, blockSize=15)
print('Computing disparity map using block matching...')
disparityBM = stereoBM.compute(aloe_l_gray, aloe_r_gray)
print('...done')

# N, XX, YY = aloe_l.shape[::-1]  # pobranie rozmiarow obrazka (kolorowego)
# visRectify = np.zeros((YY, XX*2, N), np.uint8)  # utworzenie nowego obrazka oszerokosci x2
# visRectify[:, 0:XX:, :] = aloe_l      # przypisanie obrazka lewego
# visRectify[:, XX:XX*2:, :] = aloe_r   # przypisanie obrazka prawego
#
# # Wyrysowanie poziomych linii
# for y in range(0, YY, 10):
#     cv2.line(visRectify, (0, y), (XX*2, y), (255, 0, 0))
# cv2.imshow('visRectify', visRectify)  # wizualizacja
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.figure(1)
plt.imshow(disparityBM, cmap='gray')
plt.title('BM')

# disparity range is tuned for 'aloe' image pair
window_size = 3
min_disp = 16
num_disp = 112 - min_disp

# Parameter values based on OpenCV 'stereo_match.py' example
stereoSGBM = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=16,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=32
                                   )

print('\nComputing disparity map using semi global block matching...')
disparitySGBM = stereoSGBM.compute(aloe_l_gray, aloe_r_gray)
print('...done')

plt.figure(2)
plt.imshow(disparitySGBM, cmap='gray')
plt.title('SGBM')

plt.show()
