#!/usr/bin/python
import cv2
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import scipy.ndimage.filters as filters

DIR = os.path.dirname(sys.argv[0])
IMG_NAME1 = 'fontanna1.jpg'
IMG_NAME2 = 'fontanna_pow.jpg'


def pyramid(image, blur_nbr, k, sigma):
    res_shape = (blur_nbr, image.shape[0], image.shape[1])
    res_img = np.zeros(res_shape, dtype=np.float64)
    fimage = np.float64(image)
    prev_img = cv2.GaussianBlur(fimage, (0, 0), sigmaX=sigma, sigmaY=sigma)

    for i in range(1, blur_nbr+1):
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=k**i*sigma, sigmaY=k**i*sigma)
        diff = np.float64(blurred) - np.float64(prev_img)
        diff = diff/np.max(diff)
        res_img[i-1, :, :] = diff
        prev_img = blurred

    return res_img


# TODO: rewrite to find all extremas (max and min)
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
    plt.show()


# Loading images
img1_color = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME1)
img2_color = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME2)

img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
img2_color = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)

img1 = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME2, cv2.IMREAD_GRAYSCALE)

p1 = pyramid(img1, 5, 1.26, 1.6)
e1 = find_max(p1, 7, 0.5)

p2 = pyramid(img2, 10, 1.26, 1.6)
e2 = find_max(p2, 7, 0.5)


for i in range(5):
    plt.figure()
    plt.imshow(p1[i, :, :], cmap='gray')
plt.show()
