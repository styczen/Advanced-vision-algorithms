import cv2

image = cv2.imread('lena.png')
cv2.imshow('Original', image)

gaussian_blur = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=1, sigmaY=1)
cv2.imshow('Gaussian blur', gaussian_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()