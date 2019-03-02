import cv2

image = cv2.imread('lena.png')
cv2.imshow('Original', image)

# Gaussian blur
gaussian_blur = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=3, sigmaY=0)
cv2.imshow('Gaussian blur', gaussian_blur)

# Sobel operator (edge detection)
lena_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sobel = cv2.Sobel(src=lena_gray, ddepth=cv2.CV_8U, dx=1, dy=1, ksize=3)
cv2.imshow('Sobel', sobel)

# Laplacian (edge detection, second derivative)
laplacian = cv2.Laplacian(src=lena_gray, ddepth=cv2.CV_8U, ksize=3)
cv2.imshow('Laplacian', laplacian)

# Median blur
median_blur = cv2.medianBlur(src=image, ksize=5)
cv2.imshow('Median blur', median_blur)

# Bilateral filter TODO: needs more understanding
bilateral_filter = cv2.bilateralFilter(src=image, d=10, sigmaColor=10, sigmaSpace=10)
cv2.imshow('Bilateral filter', bilateral_filter)

# Gabor kernels TODO: need more understanding
gabor_kernel = cv2.getGaborKernel(ksize=(5, 5), sigma=10, theta=1, lambd=1, gamma=1)
cv2.imshow('Gabor kernel', gabor_kernel)

cv2.waitKey(0)
cv2.destroyAllWindows()