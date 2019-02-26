import cv2
import scipy.misc

# Image scaling using OpenCV
image = cv2.imread('mandril.jpg')
height, width = image.shape[:2]
scale = 1.75
image_x2 = cv2.resize(image, (int(scale*height), int(scale*width)))
cv2.imshow('Big Mandril - OpenCV', image_x2)

# Image scaling using SciPy
image_x2_scipy = scipy.misc.imresize(image, 0.5)
cv2.imshow('Big Mandril - SciPy', image_x2)

cv2.waitKey(0)
cv2.destroyAllWindows()
