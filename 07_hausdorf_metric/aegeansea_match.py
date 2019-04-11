import matplotlib.pyplot as plt
import os
import cv2
import sys
import numpy as np

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

    return xy_c, x_c, y_c


def hausdorff(a, b):
    return np.maximum(hausdorff_helper(a, b), hausdorff_helper(b, a))


def hausdorff_helper(a, b):
    h = 0.0
    for pointA in a:
        shortest = np.Inf
        for pointB in b:
            d = np.sqrt(np.sum((pointA - pointB) ** 2))
            if d < shortest:
                shortest = d
        if shortest > h:
            h = shortest
    return h


#####
aegeansea_img = cv2.imread(DIR + '/plikiHausdorff/Aegeansea.jpg')
hsv_aegeansea = cv2.cvtColor(aegeansea_img, cv2.COLOR_BGR2HSV)

lower = np.array([0, 30, 0])
upper = np.array([55, 255, 255])

mask_aegeansea = cv2.inRange(hsv_aegeansea, lower, upper)

# REMEMBER TO ADD THIRD RETURN PARAM IN FRONT BECAUSE OF DIFFERENT OPENCV VERSIONS
contours_aegeansea, hierarchy = cv2.findContours(image=mask_aegeansea,
                                                 mode=cv2.RETR_TREE,  # first contour should be the longest
                                                 method=cv2.CHAIN_APPROX_NONE)

contours_aegeansea = np.array([el for el in contours_aegeansea if 15 < el.shape[0] < 3000])

image_file_name = 'c_astipalea.bmp'

astipalea_mask = cv2.imread(DIR + '/plikiHausdorff/imgs/' + image_file_name, cv2.IMREAD_GRAYSCALE)
astipalea_mask = 255 - astipalea_mask
astipalea_contour, _ = cv2.findContours(image=astipalea_mask,
                                        mode=cv2.RETR_TREE,  # first contour should be the longest
                                        method=cv2.CHAIN_APPROX_NONE)

astipalea_contour = np.array([el for el in astipalea_contour])
xy_centered_astipalea, x_c_astipalea, y_c_astipalea = normalize_contour(astipalea_contour, astipalea_mask)

min_dist = np.Inf
x_c_best, y_c_best, nr_best = 0, 0, 0

for i, c in enumerate(contours_aegeansea):
    mask = np.zeros(mask_aegeansea.shape, np.uint8)
    mask = cv2.drawContours(image=mask,
                            contours=[c],
                            contourIdx=-1,
                            color=255,
                            thickness=cv2.FILLED)

    xy_centered_current, x_center_current, y_center_current = normalize_contour([c], mask)

    dist = hausdorff(xy_centered_astipalea, xy_centered_current)

    if dist < min_dist:
        min_dist = dist
        nr_best = i
        x_c_best, y_c_best = x_center_current, y_center_current

    print('{} / {}; distance = {}'.format(i+1, len(contours_aegeansea), dist))

island_name = image_file_name.split('.')[0].split('_')[1]
cv2.putText(aegeansea_img, str(island_name), (int(x_c_best), int(y_c_best)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

cv2.namedWindow('aegeansea_img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('aegeansea_img', 600, 600)
cv2.imshow('aegeansea_img', aegeansea_img)

cv2.imwrite(DIR + '/result.png', aegeansea_img)
cv2.waitKey(0)

cv2.destroyAllWindows()










