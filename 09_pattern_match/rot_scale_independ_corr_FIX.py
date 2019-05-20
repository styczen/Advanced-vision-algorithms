import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h, h))


def highpassFilter(size):
    rows = np.cos(np.pi * np.matrix([-0.5 + x / (size[0] - 1) for x in range(size[0])]))
    cols = np.cos(np.pi * np.matrix([-0.5 + x / (size[1] - 1) for x in range(size[1])]))
    X = np.outer(rows, cols)
    return (1.0 - X) * (2.0 - X)


plt.close('all')
# DIR = "/home/bartek/learning/Advanced-vision-algorithms/09_pattern_match"
DIR = os.getcwd()
pattern_img = cv2.imread(DIR + '/obrazy_Mellin/domek_r0_64.pgm',
                         cv2.IMREAD_GRAYSCALE)
pattern_img_copy = pattern_img.copy()

test_img = cv2.imread(DIR + '/obrazy_Mellin/domek_r30.pgm',
                      cv2.IMREAD_GRAYSCALE)
pattern_img = pattern_img * hanning2D(pattern_img.shape[0])

pattern = np.zeros(test_img.shape)
pattern[0:pattern_img.shape[0], 0:pattern_img.shape[1]] = pattern_img
offset = int((test_img.shape[0]-pattern_img.shape[0])//2.0)
pattern_c = np.zeros(test_img.shape)
pattern_c[offset:pattern_img.shape[0]+offset,
          offset:pattern_img.shape[1]+offset] = pattern_img

# 2D fft
fft_pattern = np.fft.fft2(pattern)
fft_pattern_sh = np.fft.fftshift(fft_pattern)

fft_test = np.fft.fft2(test_img)
fft_test_sh = np.fft.fftshift(fft_test)

# Amplituda i filtracja
highpass_filter = highpassFilter(pattern.shape)
fft_pattern_abs = np.abs(fft_pattern_sh)*highpass_filter
fft_domek_abs = np.abs(fft_test_sh)*highpass_filter

# logPolar
R = fft_pattern.shape[0] // 2.0
M = 2 * R / np.log(R)
pattern_polar = cv2.logPolar(src=fft_pattern_abs,
                             center=(R, R),
                             M=M,
                             flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
domek_polar = cv2.logPolar(src=fft_domek_abs,
                           center=(R, R),
                           M=M,
                           flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

# 2D fft
fft_pattern_polar = np.fft.fft2(pattern_polar)
fft_domek_polar = np.fft.fft2(domek_polar)

fft_pattern_polar = np.fft.fftshift(fft_pattern_polar)
fft_domek_polar = np.fft.fftshift(fft_domek_polar)

# phase corr
R = np.conj(fft_pattern_polar) * fft_domek_polar
R = R / np.abs(R)
correl_norm = np.fft.ifft2(R)

# rot, scale
wsp_kata, wsp_logr = np.unravel_index(np.argmax(np.abs(correl_norm)),
                                      np.abs(correl_norm).shape)
# rozmiar_logr x rozmiar_kata to rozmiar obrazu logpolar
rozmiar_logr = domek_polar.shape[1]
rozmiar_kata = domek_polar.shape[0]
if wsp_logr > rozmiar_logr // 2:
    wykl = rozmiar_logr - wsp_logr  # powiekszenie
else:
    wykl = - wsp_logr  # pomniejszenie
A = (wsp_kata * 360.0)/rozmiar_kata
# gdzie M to parametr funkcji cv2.logPolar, a wykl wyliczamy jako:
scale = np.exp(wykl / M)
kat1 = -A  # gdzie A = (wsp_kata * 360.0 ) /rozmiar_kata
kat2 = 180 - A

srodekTrans = ((pattern_c.shape[0] / 2), (pattern_c.shape[1] / 2))
macierz_translacji_1 = cv2.getRotationMatrix2D((srodekTrans[0],
                                                srodekTrans[1]), kat1, scale)
macierz_translacji_2 = cv2.getRotationMatrix2D((srodekTrans[0],
                                                srodekTrans[1]), kat2, scale)

im_rot_scaled_1 = cv2.warpAffine(pattern_c, macierz_translacji_1,
                                 pattern.shape)
im_rot_scaled_2 = cv2.warpAffine(pattern_c, macierz_translacji_2,
                                 pattern.shape)

# FFT2
im_rot_scaled_fft_1 = np.fft.fft2(im_rot_scaled_1)
im_rot_scaled_fft_2 = np.fft.fft2(im_rot_scaled_2)

im_rot_scaled_fft_1 = np.fft.fftshift(im_rot_scaled_fft_1)
im_rot_scaled_fft_2 = np.fft.fftshift(im_rot_scaled_fft_2)

# Phase correlation
R_1 = np.conj(im_rot_scaled_fft_1)*fft_test
R_1 = R_1 / np.abs(R_1)
ccor_norm_1 = np.fft.ifft2(R_1)

R_2 = np.conj(im_rot_scaled_fft_2)*fft_test
R_2 = R_2 / np.abs(R_2)
ccor_norm_2 = np.fft.ifft2(R_2)

y_norm_1, x_norm_1 = np.unravel_index(np.argmax(np.abs(ccor_norm_1)),
                                      np.abs(ccor_norm_1).shape)
y_norm_2, x_norm_2 = np.unravel_index(np.argmax(np.abs(ccor_norm_2)),
                                      np.abs(ccor_norm_2).shape)

if np.max(np.abs(ccor_norm_1)) > np.max(np.abs(ccor_norm_2)):
    x_best = x_norm_1
    y_best = y_norm_1
    kat_best = kat1
    pattern_best = im_rot_scaled_1
    corr_best = ccor_norm_1
    macierz_translacji_best = macierz_translacji_1
else:
    x_best = x_norm_2
    y_best = y_norm_2
    kat_best = kat2
    pattern_best = im_rot_scaled_2
    corr_best = ccor_norm_2
    macierz_translacji_best = macierz_translacji_2

M = cv2.getRotationMatrix2D((test_img.shape[0]//2,
                             test_img.shape[1]//2), -kat1, 1)
test_img = cv2.warpAffine(test_img, M,
                          test_img.shape)

if abs(kat_best) < 1:
    y_best = y_best - 128
    M = np.float32([[1, 0, -x_best], [0, 1, -y_best]])
    test_img = cv2.warpAffine(test_img, M,
                              test_img.shape)

plt.figure()
# plt.imshow(pattern_img_copy, cmap='gray')
plt.imshow(im_rot_scaled_1, cmap='gray')

plt.figure()
plt.imshow(test_img, 'gray')

plt.figure()
# plt.imshow(np.abs(ccor_norm_1))
plt.imshow(np.abs(ccor_norm_1))

# print('x = {}\ny = {}\nangle = {}'.format(x_best, y_best, kat_best))

plt.show()
