import cv2

I = cv2.imread('mandril.jpg')

IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV_FULL)

cv2.imshow('gray', IG)
cv2.imshow('hsv', IHSV)

IH = IHSV[:, :, 0]
IS = IHSV[:, :, 1]
IV = IHSV[:, :, 2]

cv2.imshow('h', IH)
cv2.imshow('s', IS)
cv2.imshow('v', IV)

cv2.waitKey(0)
cv2.destroyAllWindows()

