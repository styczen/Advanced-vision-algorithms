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


def normalize_aegeansea_contours(aeg_contours, img_mask):
    normalized_aeg_contours = []
    x_centers = []
    y_centers = []
    for con in aeg_contours:
        mask_con = np.zeros(img_mask.shape, np.uint8)
        mask_con = cv2.drawContours(image=mask_con,
                                    contours=[con],
                                    contourIdx=-1,
                                    color=255,
                                    thickness=cv2.FILLED)
        xy_centered, x_c, y_c = normalize_contour([con], mask_con)
        normalized_aeg_contours.append(xy_centered)
        x_centers.append(x_c)
        y_centers.append(y_c)

    normalized_aeg_contours = np.array(normalized_aeg_contours)
    x_centers = np.array(x_centers)
    y_centers = np.array(y_centers)
    return normalized_aeg_contours, x_centers, y_centers


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

print('Normalizing Aegeansea contours...')
normalized_aeg_con, x_c_aeg, y_c_aeg = normalize_aegeansea_contours(contours_aegeansea, mask_aegeansea)
print('...done')

images = os.listdir(DIR + '/plikiHausdorff/imgs/')

for i, image_file_name in enumerate(images):
    print('\n{} / {}; image: {}'.format(i+1, len(images), image_file_name))
    image_mask = cv2.imread(DIR + '/plikiHausdorff/imgs/' + image_file_name, cv2.IMREAD_GRAYSCALE)
    image_mask = 255 - image_mask
    image_contour, _ = cv2.findContours(image=image_mask,
                                        mode=cv2.RETR_TREE,  # first contour should be the longest
                                        method=cv2.CHAIN_APPROX_NONE)

    image_contour = np.array([el for el in image_contour])
    xy_centered_image, x_c_image, y_c_image = normalize_contour(image_contour, image_mask)

    min_dist = np.Inf
    x_c_best, y_c_best = 0, 0

    for j, c in enumerate(normalized_aeg_con):
        dist = hausdorff(c, xy_centered_image)

        if dist < min_dist:
            min_dist = dist
            x_c_best, y_c_best = x_c_aeg[j], y_c_aeg[j]

        print('\t{} / {}; distance = {}'.format(j+1, len(contours_aegeansea), dist))

    island_name = image_file_name.split('.')[0].split('_')[1]

    cv2.putText(aegeansea_img, str(island_name), (int(x_c_best), int(y_c_best)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cv2.imwrite(DIR + '/result_multiple_islands.png', aegeansea_img)

cv2.namedWindow('aegeansea_img', cv2.WINDOW_NORMAL)
cv2.resizeWindow('aegeansea_img', 600, 600)
cv2.imshow('aegeansea_img', aegeansea_img)

# cv2.imwrite(DIR + '/result.png', aegeansea_img)
cv2.waitKey(0)

cv2.destroyAllWindows()
