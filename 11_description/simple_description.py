#!/usr/bin/python3
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

# Import function to plot matches from two images
import pm

DIR = os.path.dirname(sys.argv[0])

# IMG_NAME1 = 'eiffel1.jpg'
# IMG_NAME2 = 'eiffel2.jpg'

IMG_NAME1 = 'fontanna1.jpg'
IMG_NAME2 = 'fontanna2.jpg'

# IMG_NAME1 = 'budynek1.jpg'
# IMG_NAME2 = 'budynek2.jpg'

# IMG_NAME1 = 'fontanna1.jpg'
# IMG_NAME2 = 'fontanna_pow.jpg'

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
    pts = list(filter(lambda pt: pt[0] >= size and pt[0] < Y-size and
                      pt[1] >= size and pt[1] < X-size, zip(pts[0], pts[1])))
    l_otoczen = []
    l_wspolrzednych = []
    for point in pts:
        neigh = img[point[0]-size//2:point[0]+size //
                    2+1, point[1]-size//2:point[1]+size//2+1]
        w = neigh.flatten()
        w_aff = (w - np.mean(w)) / np.std(w)
        l_otoczen.append(w_aff)
        l_wspolrzednych.append(point)

    result = list(zip(l_otoczen, l_wspolrzednych))
    return result


def compare(pts1, pts2, n=20, comp_type='euclidean'):
    if comp_type == 'euclidean':
        res = compare_euclidean(pts1, pts2, n)
    elif comp_type == 'similarity':
        res = compare_similarity(pts1, pts2, n)
    else:
        print('Comparison type nor supported.')
        res = None
    return res


def compare_similarity(pts1, pts2, n=20):
    lst_pts = []
    lst_dist = []
    for neigh_tuple1 in pts1:
        closest_pts_coords = None
        closest_neight_similarity = -1
        for neigh_tuple2 in pts2:
            dot_prod = np.dot(neigh_tuple1[0], neigh_tuple2[0])
            norm_a = np.sqrt(np.dot(neigh_tuple1[0], neigh_tuple1[0]))
            norm_b = np.sqrt(np.dot(neigh_tuple2[0], neigh_tuple2[0]))
            similarity = dot_prod / (norm_a * norm_b)
            if similarity > closest_neight_similarity:
                closest_neight_similarity = similarity
                closest_pts_coords = neigh_tuple2[1]
        lst_pts.append([neigh_tuple1[1], closest_pts_coords])
        lst_dist.append(closest_neight_similarity)

    result = list(zip(lst_pts, lst_dist))
    result.sort(key=lambda x: x[1], reverse=True)

    return result[:n]


def compare_euclidean(pts1, pts2, n=20):
    lst_pts = []
    lst_dist = []
    for neigh_tuple1 in pts1:
        closest_pts_coords = None
        closest_neight_dist = np.Inf
        for neigh_tuple2 in pts2:
            dist = np.sqrt(sum(np.square(neigh_tuple1[0] - neigh_tuple2[0])))
            if dist < closest_neight_dist:
                closest_neight_dist = dist
                closest_pts_coords = neigh_tuple2[1]
        lst_pts.append([neigh_tuple1[1], closest_pts_coords])
        lst_dist.append(closest_neight_dist)

    result = list(zip(lst_pts, lst_dist))
    result.sort(key=lambda x: x[1])

    return result[:n]


plt.close('all')

# Loading images
img1 = cv2.imread(DIR + '/sift/' + IMG_NAME1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(DIR + '/sift/' + IMG_NAME2, cv2.IMREAD_GRAYSCALE)

# Find maximums
harris1 = h_fun(img1, KERNEL_SIZE)
pts1 = find_max(harris1, KERNEL_SIZE, THRESHOLD)

harris2 = h_fun(img2, KERNEL_SIZE)
pts2 = find_max(harris2, KERNEL_SIZE, THRESHOLD)

# Obtain neighbourhoods
pts_with_neigh1 = pts_description(img1, pts1, 15)
pts_with_neigh2 = pts_description(img2, pts2, 15)

# Get best matches
matches = compare(pts_with_neigh1, pts_with_neigh2, 20, 'euclidean')

pm.plot_matches(img1, img2, matches)

# mat = list(zip(*list(zip(*matches))[0]))
# x = [el[0] for el in mat[0]]
# y = [el[1] for el in mat[0]]
# pts1_xy = [x, y]

# x = [el[0] for el in mat[1]]
# y = [el[1] for el in mat[1]]
# pts2_xy = [x, y]

# draw_points(img1, pts1_xy)
# draw_points(img2, pts2_xy)

# draw_points(img1, pts1)
# draw_points(img2, pts2)

plt.show()
