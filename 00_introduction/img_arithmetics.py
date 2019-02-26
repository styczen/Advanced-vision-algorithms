import cv2
import numpy as np

mandril = cv2.imread('mandril.jpg')
lena = cv2.imread('lena.png')

mandril = cv2.cvtColor(mandril, cv2.COLOR_BGR2GRAY)
lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
cv2.imshow('Lena', lena)
cv2.imshow('Mandril', mandril)

# Adding
add = mandril + lena
cv2.imshow('Mandril + Lena', add)

# Subtracting
subtract = np.int16(mandril) - np.int16(lena)
subtract = abs(subtract)
subtract = np.uint8(subtract)
cv2.imshow('Mandril - Lena', subtract)

subtract_opencv = cv2.absdiff(mandril, lena)
cv2.imshow('absdiff(Mandril, Lena)', subtract_opencv)

# Multiplying
multiply = mandril * lena
cv2.imshow('Mandril * Lena', multiply)

# Linear combination 
lin_comb = 0.3*mandril + 0.5*lena
cv2.imshow('Linear combination', np.uint8(lin_comb))


cv2.waitKey(0)
cv2.destroyAllWindows()