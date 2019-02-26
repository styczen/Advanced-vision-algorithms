import cv2

image = cv2.imread('mandril.jpg')
height, width = image.shape[:2]
scale = 1.75
image_x2 = cv2.resize(image, (int(scale*height), int(scale*width)))
cv2.imshow('Big Mandril', image_x2)
cv2.waitKey(0)

cv2.destroyAllWindows()
