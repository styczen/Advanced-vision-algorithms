#!/usr/bin/python3
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

DIR = os.path.dirname(sys.argv[0])

# IMG_NAME1 = 'eiffel1.jpg'
# IMG_NAME2 = 'eiffel2.jpg'

# IMG_NAME1 = 'fontanna1.jpg'
# IMG_NAME2 = 'fontanna2.jpg'

# IMG_NAME1 = 'budynek1.jpg'
# IMG_NAME2 = 'budynek2.jpg'

IMG_NAME1 = 'fontanna1.jpg'
IMG_NAME2 = 'fontanna_pow.jpg'

# Loading images
img1 = cv2.imread(DIR + '/sift/' + IMG_NAME1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(DIR + '/sift/' + IMG_NAME2, cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
kp1, desc1 = sift.detectAndCompute(image=img1, mask=None)
kp2, desc2 = sift.detectAndCompute(image=img2, mask=None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

matches = [[m] for m, n in matches if m.distance < 0.15*n.distance]
outImg = cv2.drawMatchesKnn(
    img1, kp1, img2, kp2, matches, outImg=None, flags=2)

plt.figure()
plt.imshow(outImg)
plt.show()
