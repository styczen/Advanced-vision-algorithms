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
input_img = cv2.imread(DIR + '/plikiHausdorff/ithaca_query.bmp')

mask = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
input_img = 255 - input_img
mask = 255 - mask

# REMEMBER TO ADD THIRD RETURN PARAM IN FRONT BECAUSE OF DIFFERENT OPENCV VERSIONS
contours, hierarchy = cv2.findContours(image=mask,
                                       mode=cv2.RETR_TREE,  # first contour should be the longest
                                       method=cv2.CHAIN_APPROX_NONE)

img_contours = cv2.drawContours(image=input_img,
                                contours=contours,
                                contourIdx=0,
                                color=(255, 0, 0))

xy_centered_ithaca, x_center_ithaca, y_center_ithaca = normalize_contour(contours, mask)

images = os.listdir(DIR + '/plikiHausdorff/imgs/')
min_dist_name = ''
min_dist = np.Inf
for i, image_name in enumerate(images):
    input_img = cv2.imread(DIR + '/plikiHausdorff/imgs/' + image_name)
    mask = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    mask = 255 - mask
    contours, hierarchy = cv2.findContours(image=mask,
                                           mode=cv2.RETR_TREE,  # first contour should be the longest
                                           method=cv2.CHAIN_APPROX_NONE)

    xy_centered_current, x_center_current, y_center_current = normalize_contour(contours, mask)

    dist = hausdorff(xy_centered_ithaca, xy_centered_current)

    if dist < min_dist:
        min_dist = dist
        min_dist_name = image_name

    print('{} / {}; distance = {}; image: {}'.format(i+1, len(images), dist, image_name))

print('Minimum distance = {} for {}'.format(min_dist, min_dist_name))



# plt.figure()
# plt.imshow(img_contours)
#
# plt.show()


# cv2.imshow('kek', img_contours)
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()
