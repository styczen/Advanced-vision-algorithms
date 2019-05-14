#!/usr/bin/python
import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import scipy.ndimage.filters as filters

DIR = os.path.abspath('')

IMG_NAME1 = 'fontanna1.jpg'
IMG_NAME2 = 'fontanna2.jpg'

# IMG_NAME1 = 'budynek1.jpg'
# IMG_NAME2 = 'budynek2.jpg'

KERNEL_SIZE = 7
THRESHOLD = 0.2


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
    plt.imshow(img)
    plt.plot(points[1], points[0], '*', color='r')
    # plt.show()


# Loading images
img1_color = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME1)
img2_color = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME2)

img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
img2_color = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)

img1 = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME2, cv2.IMREAD_GRAYSCALE)

# Find maximums
h1 = h_fun(img1, KERNEL_SIZE)
m1 = find_max(h1, KERNEL_SIZE, THRESHOLD)

h2 = h_fun(img2, KERNEL_SIZE)
m2 = find_max(h2, KERNEL_SIZE, THRESHOLD)

draw_points(img1_color, m1)
draw_points(img2_color, m2)
plt.show()
