import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h, h))


def highpassFilter(size):
    rows = np.cos(np.pi*np.matrix([-0.5 + x/(size[0]-1) for x in range(size[0])]))
    cols = np.cos(np.pi*np.matrix([-0.5 + x/(size[1]-1) for x in range(size[1])]))
    X = np.outer(rows, cols)
    return (1.0 - X)*(2.0 - X)


DIR = os.path.dirname(sys.argv[0])


print('Loading images...')
pattern = cv2.imread(DIR + '/obrazy_Mellin/domek_r0_64.pgm', cv2.IMREAD_GRAYSCALE)
test_img = cv2.imread(DIR + '/obrazy_Mellin/domek_r0.pgm', cv2.IMREAD_GRAYSCALE)

# Load color version of test image for more user friendly view of results
test_img_color = cv2.imread(DIR + '/obrazy_Mellin/domek_r30.pgm')
print('...done')

# Multiply pattern by hanning window
pattern_win = pattern * hanning2D(pattern.shape[0])

# Expand pattern to match test image size filling it with zeros
ext_pattern = np.zeros(test_img.shape, dtype=np.uint8)
ext_pattern[0:pattern_win.shape[0], 0:pattern_win.shape[1]] = pattern_win

# For display purposes expand pattern without hanning window applied
ext_pattern1 = np.zeros(test_img.shape, dtype=np.uint8)
ext_pattern1[0:pattern.shape[0], 0:pattern.shape[1]] = pattern

# FFT
ext_pattern_fft = np.fft.fft2(ext_pattern)
ext_pattern_fft = np.fft.fftshift(ext_pattern_fft)

test_img_fft = np.fft.fft2(test_img)
test_img_fft = np.fft.fftshift(test_img_fft)

# Amplitude
ext_pattern_fft_abs = np.abs(ext_pattern_fft)
test_img_fft_abs = np.abs(test_img_fft)

# Highpass filtration
ext_pattern_filtered = ext_pattern_fft_abs * highpassFilter(ext_pattern_fft_abs.shape)
test_img_filtered = test_img_fft_abs * highpassFilter(test_img_fft_abs.shape)

# Log-polar transform
# Pattern
middle_point = (ext_pattern_filtered.shape[0]//2, ext_pattern_filtered.shape[1]//2)
M = 2 * ext_pattern_filtered.shape[0] / np.log(ext_pattern_filtered.shape[0])
ext_pattern_logpolar = cv2.logPolar(ext_pattern_filtered, middle_point, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

# Test image
middle_point = (test_img_filtered.shape[0]//2, test_img_filtered.shape[1]//2)
M = 2 * test_img_filtered.shape[0] / np.log(test_img_filtered.shape[0])
test_img_logpolar = cv2.logPolar(test_img_filtered, middle_point, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

# Log-polar FFT
ext_pattern_lp_fft = np.fft.fft2(ext_pattern_logpolar)
ext_pattern_lp_fft = np.fft.fftshift(ext_pattern_lp_fft)

test_img_lp_fft = np.fft.fft2(test_img_logpolar)
test_img_lp_fft = np.fft.fftshift(test_img_lp_fft)

# Phase correlation
R = np.conj(ext_pattern_lp_fft) * test_img_lp_fft
R = R / np.abs(R)
ccor_norm = np.fft.ifft2(R)
ccor_norm_abs = np.abs(ccor_norm)

# Maximum in log-polar
wsp_logr, wsp_kata = np.unravel_index(np.argmax(ccor_norm_abs), ccor_norm_abs.shape)

# Calculate scaling and rotation from log-polar max
rozmiar_logr = ext_pattern_logpolar.shape[0]
rozmiar_kata = ext_pattern_logpolar.shape[1]

if wsp_logr > rozmiar_logr//2:  # rozmiar_logr x rozmiar_kata to rozmiar obrazu logpolar
    wykl = rozmiar_logr - wsp_logr  # powiekszenie
else:
    wykl = - wsp_logr               # pomniejszenie

skala = np.exp(wykl / M)   # gdzie M to parametr funkcji cv2.logPolar, a wykl wyliczamy jako:
A = (wsp_kata * 360.0) / rozmiar_kata
kat1 = - A
kat2 = 180 - A

# Affine transform for both angles
im = np.zeros(test_img.shape, dtype=np.uint8)
middle = (im.shape[0]//2, im.shape[1]//2)
pattern_middle = (pattern.shape[0]//2, pattern.shape[1]//2)
im[middle[0]-pattern_middle[0]:middle[0]+pattern_middle[0], middle[1]-pattern_middle[1]:middle[1]+pattern_middle[1]] \
    = pattern

srodekTrans = (im.shape[0] / 2 - 0.5, im.shape[1] / 2 - 0.5)  # im to obraz wzorca uzupelniony zerami,
# ale ze wzorcem umieszczonymna srodku, a nie w lewym, gornym rogu!
macierz_translacji1 = cv2.getRotationMatrix2D(srodekTrans, kat1, skala)
macierz_translacji2 = cv2.getRotationMatrix2D(srodekTrans, kat2, skala)

obraz_obrocony_przeskalowany1 = cv2.warpAffine(im, macierz_translacji1, ext_pattern.shape)
obraz_obrocony_przeskalowany2 = cv2.warpAffine(im, macierz_translacji2, ext_pattern.shape)

# FFT for transform imgs for both angles
obraz_obrocony_przeskalowany1_fft = np.fft.fft2(obraz_obrocony_przeskalowany1)
# obraz_obrocony_przeskalowany1_fft = np.fft.fftshift(obraz_obrocony_przeskalowany1_fft)

obraz_obrocony_przeskalowany2_fft = np.fft.fft2(obraz_obrocony_przeskalowany2)
# obraz_obrocony_przeskalowany2_fft = np.fft.fftshift(obraz_obrocony_przeskalowany2_fft)

# Phase correlation with test image
R = np.conj(obraz_obrocony_przeskalowany1_fft) * test_img_fft
R = R / np.abs(R)
ccor_norm = np.fft.ifft2(R)
ccor_norm_abs1 = np.abs(ccor_norm)

R = np.conj(obraz_obrocony_przeskalowany2_fft) * test_img_fft
R = R / np.abs(R)
ccor_norm = np.fft.ifft2(R)
ccor_norm_abs2 = np.abs(ccor_norm)

# Find greater correlation between both angles
if np.max(ccor_norm_abs1) >= np.max(ccor_norm_abs2):
    y, x = np.unravel_index(np.argmax(ccor_norm_abs1), ccor_norm_abs1.shape)
else:
    y, x = np.unravel_index(np.argmax(ccor_norm_abs2), ccor_norm_abs2.shape)

# Draw found maximum
cv2.circle(img=test_img_color,
           center=(x, y),
           radius=1,
           color=(0, 255, 0),
           thickness=3)

# Transform pattern to match the place in image from test image
tr_m = np.float32([[1, 0, x], [0, 1, y]])
ext_pattern_tr = cv2.warpAffine(ext_pattern, tr_m, (ext_pattern.shape[1], ext_pattern.shape[0]))

plt.figure()
plt.imshow(ext_pattern_tr, cmap='gray')

plt.figure()
plt.imshow(test_img_color)
plt.show()
