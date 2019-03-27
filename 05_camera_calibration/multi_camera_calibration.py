import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# if __name__ == '__main__':
DIR = os.getcwd()
DIR = DIR + '/calibration_stereo/'

#  kryteria przetwania obliczen (blad+liczba iteracji)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# przygotowanie punktow 2D w postaci: (0,0,0), (1,0,0), (2,0,0) ....,(6,7,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# tablice do przechowywania punktow obiektow (3D) i punktow na obrazie (2D) dla wszystkich obrazow
objpoints = []  # punkty 3d w przestrzeni (rzeczywsite)

imgpoints_l = []  # punkty 2d w plaszczyznie obrazu.
imgpoints_r = []  # punkty 2d w plaszczyznie obrazu.

for fname in range(1, 13):
    # wczytanie obrazu
    img_l = cv2.imread(DIR + 'images_left/left%02d.jpg' % fname)
    img_r = cv2.imread(DIR + 'images_right/right%02d.jpg' % fname)

    # konwersja do odcieni szarosci
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    # wyszukiwanie naroznikow na planszy
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (7, 6), None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (7, 6), None)

    # jesli znaleniono na obrazie punkty
    if ret_l is True and ret_r is True:
        # dolaczenie wspolrzednych 3D
        objpoints.append(objp)

        # poprawa lokalizacji punktow (podpiskelowo)
        corners2_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)

        # dolaczenie poprawionych punktow
        imgpoints_l.append(corners2_l)

        # poprawa lokalizacji punktow (podpiskelowo)
        corners2_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        # dolaczenie poprawionych punktow
        imgpoints_r.append(corners2_r)

ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, gray_l.shape[::-1], None, None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, gray_r.shape[::-1], None, None)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                                                                 imgpoints_l,
                                                                                                 imgpoints_r,
                                                                                                 mtx_l,
                                                                                                 dist_l,
                                                                                                 mtx_r,
                                                                                                 dist_r,
                                                                                                 gray_l.shape[::-1])

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2,
                                                                  distCoeffs2, gray_r.shape[::-1], R, T)

map1_L, map2_L = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, gray_l.shape[::-1], cv2.CV_16SC2)
map1_R, map2_R = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, gray_r.shape[::-1], cv2.CV_16SC2)

dst_L = cv2.remap(img_l, map1_L, map2_L, cv2.INTER_LINEAR)
dst_R = cv2.remap(img_r, map1_R, map2_R, cv2.INTER_LINEAR)

N, XX, YY = dst_L.shape[::-1]  # pobranie rozmiarow obrazka (kolorowego)
visRectify = np.zeros((YY, XX*2, N), np.uint8)  # utworzenie nowego obrazka oszerokosci x2
visRectify[:, 0:640:, :] = dst_L      # przypisanie obrazka lewego
visRectify[:, 640:1280:, :] = dst_R   # przypisanie obrazka prawego

# Wyrysowanie poziomych linii
for y in range(0, 480, 10):
    cv2.line(visRectify, (0, y), (1280, y), (255, 0, 0))
cv2.imshow('visRectify', visRectify)  # wizualizacja

cv2.waitKey(0)
cv2.destroyAllWindows()
