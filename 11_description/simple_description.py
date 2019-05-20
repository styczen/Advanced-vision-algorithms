#!/usr/bin/python
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

DIR = os.path.dirname(sys.argv[0])
IMG_NAME1 = 'eiffel1.jpg'
IMG_NAME2 = 'eiffel2.jpg'
KERNEL_SIZE = 7
THRESHOLD = 0.2
NEIGHBOURHOOD = 15


def h_fun(img, kernel_size=3):
    """Calculates Harris operator array for every pixel"""
    Ix = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=kernel_size)
    Iy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=kernel_size)
    Ix_square = Ix * Ix
    Iy_square = Iy * Iy
    Ixy = Ix * Iy
    Ix_square_blur = cv2.GaussianBlur(Ix_square, (kernel_size, kernel_size), 0)
    Iy_square_blur = cv2.GaussianBlur(Iy_square, (kernel_size, kernel_size), 0)
    Ixy_blur = cv2.GaussianBlur(Ixy, (kernel_size, kernel_size), 0)
    det = Ix_square_blur * Iy_square_blur - Ixy_blur * Ixy_blur
    trace = Ix_square_blur + Iy_square_blur
    k = 0.05
    h = det - k*trace*trace
    h = h / np.max(h)
    return h


def find_max(image, size, threshold):
    """Finds maximum of array"""
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = (image > threshold)
    maxima[diff == 0] = 0
    return np.nonzero(maxima)


def draw_points(img, points):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.plot(points[1], points[0], '*', color='r')


def pts_description(img, pts, size):
    X, Y = img.shape[:2]  # X - height, Y - width
    pts = list(filter(lambda pt: pt[0]>=size and pt[0]<Y-size and pt[1]>=size and pt[1]<X-size, zip(pts[0], pts[1])))
    l_otoczen = []
    l_wspolrzednych = []
    for point in pts:
        neigh = img[point[0]-size//2:point[0]+size//2+1, point[1]-size//2:point[1]+size//2+1]
        l_otoczen.append(neigh.flatten())
        l_wspolrzednych.append(point)

    result = list(zip(l_otoczen, l_wspolrzednych))
    return result


def compare(pts1, pts2, n):
    best_neight = []
    for neigh_tuple1 in pts1:
        closest_pts_coords = None
        closest_neight_dist = np.Inf
        for neigh_tuple2 in pts2:
            dist = np.sqrt(np.square(neigh_tuple1[0]) + np.square(neigh_tuple2[0]))
            if dist < closest_neight_dist:
                closest_neight_dist = dist
                closest_pts_coords = neigh_tuple2[1]
        best_neight.append(closest_pts_coords )



# Loading images
# img1_color = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME1)
# img2_color = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME2)
#
# img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
# img2_color = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)

img1 = cv2.imread(DIR + '/sift/' + IMG_NAME1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(DIR + '/sift/' + IMG_NAME2, cv2.IMREAD_GRAYSCALE)

# Find maximums
harris1 = h_fun(img1, KERNEL_SIZE)
pts1 = find_max(harris1, KERNEL_SIZE, THRESHOLD)

harris2 = h_fun(img2, KERNEL_SIZE)
pts2 = find_max(harris2, KERNEL_SIZE, THRESHOLD)

# Obtain neighbourhoods
pts_with_neigh1 = pts_description(img1, pts1, NEIGHBOURHOOD)
pts_with_neigh2 = pts_description(img2, pts2, NEIGHBOURHOOD)

draw_points(img1, pts1)
draw_points(img2, pts2)
plt.show()
