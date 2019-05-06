import cv2
import numpy as np
import os
import sys

DIR = os.path.dirname(sys.argv[0])
DELTA = 30

print('Loading template...')
trybik = cv2.imread(DIR + '/obrazy_hough/trybik.jpg')
trybik_gray = cv2.cvtColor(trybik, cv2.COLOR_BGR2GRAY)
trybik_gray = 255 - trybik_gray
print('...done\n')

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
                          contours=[contours[0]],
                          contourIdx=0,
                          color=(255, 0, 0))

cv2.circle(img=trybik,
           center=(np.int(x_c), np.int(y_c)),
           radius=2,
           color=(0, 255, 0),
           thickness=2)

print('Calculating R-table...')
sobelx = cv2.Sobel(trybik_gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(trybik_gray, cv2.CV_64F, 0, 1, ksize=5)

gradient = np.sqrt(sobelx ** 2 + sobely ** 2)
gradient = gradient / gradient.max()

orientation = np.rad2deg(np.arctan2(sobely, sobelx))
orientation += 180
orientation = np.uint16(orientation)

Rtable = [[] for _ in range(360)]

for point in contours[0]:
    omega = orientation[point[0, 1], point[0, 0]]
    r = np.sqrt((point[0, 0] - x_c) ** 2 + (point[0, 1] - y_c) ** 2)
    beta = np.arctan2(point[0, 1] - y_c, point[0, 0] - x_c)
    if omega == 360:
        omega = 0
    Rtable[omega].append([r, beta])
print('...done\n')

# %%
print('Loading test image...')
trybik2 = cv2.imread(DIR + '/obrazy_hough/trybiki2.jpg')
trybik2_gray = cv2.cvtColor(trybik2, cv2.COLOR_BGR2GRAY)
print('...done\n')

sobelx2 = cv2.Sobel(trybik2_gray, cv2.CV_64F, 1, 0, ksize=5)
sobely2 = cv2.Sobel(trybik2_gray, cv2.CV_64F, 0, 1, ksize=5)

gradient2 = np.sqrt(sobelx2 ** 2 + sobely2 ** 2)
gradient2 = gradient2 / gradient2.max()

orientation2 = np.rad2deg(np.arctan2(sobely2, sobelx2))
orientation2 += 180
orientation2 = np.uint16(orientation2)

# accum = np.zeros(trybik2.shape[:2], dtype=np.uint8)

nr_angles = 10
new_hough_shape = trybik2.shape[:2] + (nr_angles,)
new_hough = np.zeros(new_hough_shape)
angles = [angle for angle in range(0, 360, int(360 / nr_angles))]

print('Finding matches...')
for row in range(gradient2.shape[0]):
    for col in range(gradient2.shape[1]):
        if gradient2[row, col] > 0.5:
            for i in range(len(angles)):
                df = angles[i]
                table = Rtable[orientation2[row, col] - df]
                for t in table:
                    x1 = int(col + t[0]*np.cos(t[1] + np.deg2rad(df)))
                    y1 = int(row + t[0]*np.sin(t[1] + np.deg2rad(df)))
                    if 0 <= x1 < gradient2.shape[1] and 0 <= y1 < gradient2.shape[0]:
                        new_hough[y1, x1, i] += 1
print('...done\n')

print('Drawing found matches...')
for i in range(5):
    idx = np.argmax(new_hough)
    idx = np.unravel_index(idx, new_hough_shape)
    xc = idx[1]
    yc = idx[0]
    df = angles[idx[2]]
    cv2.circle(img=trybik2,
               center=(np.int(xc), np.int(yc)),
               radius=2,
               color=(0, 0, 255),
               thickness=2)
    cv2.putText(trybik2, str(i), (int(xc), int(yc)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    for omega_item in Rtable:
        for item in omega_item:
            r = item[0]
            beta = item[1]
            x = xc - r*np.cos(beta + np.deg2rad(df))
            y = yc - r*np.sin(beta + np.deg2rad(df))
            cv2.circle(img=trybik2,
                       center=(np.int(x), np.int(y)),
                       radius=1,
                       color=(255, 0, 0))
    new_hough[idx[0]-DELTA:idx[0]+DELTA, idx[1]-DELTA:idx[1]+DELTA, :] = 0
print('...done\n')

cv2.imshow('trybik2', trybik2)

cv2.waitKey(0)

cv2.destroyAllWindows()
