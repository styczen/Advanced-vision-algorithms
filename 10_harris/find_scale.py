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


def pyramid(image, blur_nbr, k, sigma):
    res_shape = (blur_nbr, image.shape[0], image.shape[1])
    res_img = np.zeros(res_shape, dtype=np.float64)
    fimage = np.float64(image)
    prev_img = cv2.GaussianBlur(fimage, (0, 0), sigmaX=sigma, sigmaY=sigma)

    for i in range(1, blur_nbr+1):
        blurred = cv2.GaussianBlur(
            image, (0, 0), sigmaX=k**i*sigma, sigmaY=k**i*sigma)
        diff = np.float64(blurred) - np.float64(prev_img)
        diff = diff/np.max(diff)
        res_img[i-1] = diff
        prev_img = blurred

    return res_img


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


def draw_pyramids(pyramid, pts, nr):
    for i in range(pyramid.shape[0]):
        x = pts[2]  # wspolrzedne x we wszystkich skalach
        x = x[pts[0] == i]  # jako wspolrzedne x tylko w skali i
        y = pts[1]  # wspolrzedne x we wszystkich skalach
        y = y[pts[0] == i]  # jako wspolrzedne x tylko w skali i
        plt.figure()
        plt.imshow(pyramid[i], 'gray')
        plt.scatter(x=x, y=y, marker='*', c='r')
        plt.title('{}, scale = {}'.format(nr, i))


# Loading images
img1_color = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME1)
img2_color = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME2)

img1_color = cv2.cvtColor(img1_color, cv2.COLOR_BGR2RGB)
img2_color = cv2.cvtColor(img2_color, cv2.COLOR_BGR2RGB)

img1 = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(DIR + '/pliki_harris/' + IMG_NAME2, cv2.IMREAD_GRAYSCALE)

p1 = pyramid(img1, 5, 1.26, 1.6)
p2 = pyramid(img2, 10, 1.26, 1.6)

e1 = find_max(p1, 7, 0.5)
e2 = find_max(p2, 7, 0.5)

draw_pyramids(p1, e1, 1)
draw_pyramids(p2, e2, 2)

# for i in range(5):
#     x = e1[2]  # wspolrzedne x we wszystkich skalach
#     x = x[e1[0] == i]  # jako wspolrzedne x tylko w skali i
#     y = e1[1]  # wspolrzedne x we wszystkich skalach
#     y = y[e1[0] == i]  # jako wspolrzedne x tylko w skali i
#     plt.figure()
#     plt.imshow(p1[i], 'gray')
#     plt.scatter(x=x, y=y, marker='*', c='r')
#     plt.title('img 1 scale='+str(i))
#     # plt.show()
# for i in range(10):
#     x = e2[2]  # wspolrzedne x we wszystkich skalach
#     x = x[e2[0] == i]  # jako wspolrzedne x tylko w skali i
#     y = e2[1]  # wspolrzedne x we wszystkich skalach
#     y = y[e2[0] == i]  # jako wspolrzedne x tylko w skali i
#     plt.figure()
#     plt.imshow(p2[i], 'gray')
#     plt.scatter(x=x, y=y, marker='*', c='r')
#     plt.title('img 2 scale='+str(i))
plt.show()

# pyramid_draw_points(p1, e1)
# pyramid_draw_points(p2, e2)
# plt.show()
# 10 scales

# # Use Harris for both two images
# h1 = h_fun(img1, KERNEL_SIZE)
# m1 = find_max(h1, KERNEL_SIZE, THRESHOLD)

# h2 = h_fun(img2, KERNEL_SIZE)
# m2 = find_max(h2, KERNEL_SIZE, THRESHOLD)

# draw_points(img1, m1)
# draw_points(img2, m2)
plt.show()
