import matplotlib.pyplot as plt
import os
import cv2
import sys
import numpy as np
from scipy.optimize import minimize, fmin

DIR = os.path.dirname(sys.argv[0])


def normalize_contour(c, img):
    xy = c[0][:, 0, :]

    m = cv2.moments(img, True)
    x_c = m['m10'] / m['m00']
    y_c = m['m01'] / m['m00']

    xy_c = xy - (x_c, y_c)

    max_dist = 0.0
    for row in xy_c:
        for row1 in xy_c:
            d = np.sqrt(np.sum((row - row1) ** 2))
            if d > max_dist:
                max_dist = d

    xy_c = np.float32(xy_c)
    xy_c = xy_c / max_dist

    return xy_c[:, 0], xy_c[:, 1], x_c, y_c


def hausdorff(angle, x1, y1, x2, y2):
    nx1 = x1 * np.cos(angle) - y1 * np.sin(angle)
    ny1 = x1 * np.sin(angle) + y1 * np.cos(angle)
    x1 = nx1
    y1 = ny1
    return np.maximum(hausdorff_helper(x1, y1, x2, y2), hausdorff_helper(x2, y2, x1, y1))


def hausdorff_helper(a1, a2, b1, b2):
    h = 0.0
    for pointA in zip(a1, a2):
        shortest = np.Inf
        for pointB in zip(b1, b2):
            d = np.sqrt(np.sum((np.subtract(pointA, pointB)) ** 2))
            if d < shortest:
                shortest = d
        if shortest > h:
            h = shortest
    return h


#####
input_img = cv2.imread(DIR + '/plikiHausdorff/ithaca_query.bmp')

mask_ithaca = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
mask_ithaca = 255 - mask_ithaca

# REMEMBER TO ADD THIRD RETURN PARAM IN FRONT BECAUSE OF DIFFERENT OPENCV VERSIONS
contours_ithaca, _ = cv2.findContours(image=mask_ithaca,
                                      mode=cv2.RETR_TREE,  # first contour should be the longest
                                      method=cv2.CHAIN_APPROX_NONE)

x_normalized_ithaca, y_normalized_ithaca, x_c_ithaca, y_c_ithaca = normalize_contour(contours_ithaca, mask_ithaca)

#####
# Load 'sa_hydra.bmp' - closest contour to ithaca_query.bmp - found in 'hausdorff_tb.py' script
img_hydra = cv2.imread(DIR + '/plikiHausdorff/imgs/sa_hydra.bmp')

mask_hydra = cv2.cvtColor(img_hydra, cv2.COLOR_BGR2GRAY)
mask_hydra = 255 - mask_hydra

# REMEMBER TO ADD THIRD RETURN PARAM IN FRONT BECAUSE OF DIFFERENT OPENCV VERSIONS
contours_hydra, _ = cv2.findContours(image=mask_hydra,
                                     mode=cv2.RETR_TREE,  # first contour should be the longest
                                     method=cv2.CHAIN_APPROX_NONE)


x_normalized_hydra, y_normalized_hydra, x_center_hydra, y_center_hydra = normalize_contour(contours_hydra, mask_hydra)

# Plot ithaca and closest (Hausdorff distance) contours
plt.figure()
plt.plot(x_normalized_ithaca, y_normalized_ithaca, 'b')
plt.plot(x_normalized_hydra, y_normalized_hydra, 'r')
plt.gca().invert_yaxis()

#####
# Load 'i_ithaca.bmp' and test Hausdorff distance with rotation
img_ithaca = cv2.imread(DIR + '/plikiHausdorff/imgs/i_ithaca.bmp')

mask_ithaca_rot = cv2.cvtColor(img_ithaca, cv2.COLOR_BGR2GRAY)
mask_ithaca_rot = 255 - mask_ithaca_rot

# REMEMBER TO ADD THIRD RETURN PARAM IN FRONT BECAUSE OF DIFFERENT OPENCV VERSIONS
contours_ithaca_rot, _ = cv2.findContours(image=mask_ithaca_rot,
                                          mode=cv2.RETR_TREE,  # first contour should be the longest
                                          method=cv2.CHAIN_APPROX_NONE)

x_norm_ithaca_rot, y_norm_ithaca_rot, x_c_ithaca_rot, y_c_ithaca_rot = normalize_contour(contours_ithaca_rot, mask_ithaca_rot)

nr_angles = 10
angleMin = np.zeros(nr_angles)
dMin = np.zeros(nr_angles)
for i in range(nr_angles):
    # angleMin[i] = np.deg2rad(360.0 / nr_angles * i)
    angleMin[i] = fmin(hausdorff, np.deg2rad(36 * i), (x_normalized_ithaca, y_normalized_ithaca, x_norm_ithaca_rot, y_norm_ithaca_rot))
    dMin[i] = hausdorff(angleMin[i], x_normalized_ithaca, y_normalized_ithaca, x_norm_ithaca_rot, y_norm_ithaca_rot)
    print('Iteration = {}; distance = {}; angle = {}'.format(i, dMin[i], angleMin[i]))
hausd = min(dMin)
min_idx = dMin.argmin()
fi = angleMin[min_idx]

print('\nBest angle = {}'.format(np.rad2deg(fi)))

# dist = hausdorff(90, x_normalized_ithaca, y_normalized_ithaca, x_norm_ithaca_rot, y_norm_ithaca_rot)
# fi = np.deg2rad(90)

nx1 = x_norm_ithaca_rot * np.cos(fi) - y_norm_ithaca_rot * np.sin(fi)
ny1 = x_norm_ithaca_rot * np.sin(fi) + y_norm_ithaca_rot * np.cos(fi)
x_norm_ithaca_rot = nx1
y_norm_ithaca_rot = ny1

# Plot ithaca and rotated 'i_ithaca.bmp'
plt.figure()
plt.plot(x_normalized_ithaca, y_normalized_ithaca, 'b')
plt.plot(x_norm_ithaca_rot, y_norm_ithaca_rot, 'r')
plt.gca().invert_yaxis()
