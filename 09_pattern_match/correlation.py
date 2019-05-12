import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

DIR = os.path.dirname(sys.argv[0])

print('Loading images...')
pattern = cv2.imread(DIR + '/obrazy_Mellin/wzor.pgm', cv2.IMREAD_GRAYSCALE)
test_img = cv2.imread(DIR + '/obrazy_Mellin/domek_r0.pgm', cv2.IMREAD_GRAYSCALE)
test_img_color = cv2.imread(DIR + '/obrazy_Mellin/domek_r0.pgm')
print('...done')

ext_pattern = np.zeros(test_img.shape, dtype=np.uint8)
ext_pattern[0:pattern.shape[0], 0:pattern.shape[1]] = pattern

ext_pattern_fft = np.fft.fft2(ext_pattern)
test_img_fft = np.fft.fft2(test_img)

# Correlation
ccor_fft = np.conj(ext_pattern_fft) * test_img_fft
ccor = np.fft.ifft2(ccor_fft)
ccor_abs = np.abs(ccor)

plt.figure()
plt.imshow(ccor_abs, cmap='gray')
plt.title('abs(ccor)')

y, x = np.unravel_index(np.argmax(ccor_abs), ccor_abs.shape)

cv2.circle(img=test_img_color,
           center=(x, y),
           radius=1,
           color=(255, 0, 0),
           thickness=3)

# Phase correlation
R = np.conj(ext_pattern_fft) * test_img_fft
R = R / np.abs(R)

ccor_norm = np.fft.ifft2(R)

ccor_norm_abs = np.abs(ccor_norm)

plt.figure()
plt.imshow(ccor_norm_abs, cmap='gray')
plt.title('abs(R)')

y, x = np.unravel_index(np.argmax(ccor_norm_abs), ccor_norm_abs.shape)

cv2.circle(img=test_img_color,
           center=(x, y),
           radius=1,
           color=(0, 255, 0),
           thickness=3)

tr_m = np.float32([[1, 0, x], [0, 1, y]])
ext_pattern_tr = cv2.warpAffine(ext_pattern, tr_m, (ext_pattern.shape[1], ext_pattern.shape[0]))

plt.figure()
plt.imshow(ext_pattern_tr, cmap='gray')

plt.figure()
plt.imshow(test_img_color)
plt.show()
