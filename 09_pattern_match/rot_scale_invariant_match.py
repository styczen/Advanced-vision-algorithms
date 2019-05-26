#!/usr/bin/python3
import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

DIR = os.path.dirname(sys.argv[0])


def hanning2D(n):
    h = np.hanning(n)
    return np.sqrt(np.outer(h, h))


def highpassFilter(size):
    rows = np.cos(
        np.pi * np.matrix([-0.5 + x / (size[0] - 1) for x in range(size[0])]))
    cols = np.cos(
        np.pi * np.matrix([-0.5 + x / (size[1] - 1) for x in range(size[1])]))
    X = np.outer(rows, cols)
    return (1.0 - X) * (2.0 - X)


pattern_img = cv2.imread(
    DIR + '/obrazy_Mellin/domek_r0_64.pgm', cv2.IMREAD_GRAYSCALE)

pattern_img_copy = pattern_img.copy()

for i in range(10, 90, 10):
    print('\nAngle = {}'.format(i))
    test_img = cv2.imread(
        DIR + '/obrazy_Mellin/domek_s{}.pgm'.format(i), cv2.IMREAD_GRAYSCALE)

    pattern_img = pattern_img * hanning2D(pattern_img.shape[0])

    pattern = np.zeros(test_img.shape)
    pattern[0:pattern_img.shape[0], 0:pattern_img.shape[1]] = pattern_img

    # 2D FFT
    fft_pattern = np.fft.fft2(pattern)
    fft_pattern_sh = np.fft.fftshift(fft_pattern)
    fft_test = np.fft.fft2(test_img)
    fft_test_sh = np.fft.fftshift(fft_test)

    # Highpass filtering
    highpass_filter = highpassFilter(pattern.shape)
    fft_pattern_abs = np.abs(fft_pattern_sh)*highpass_filter
    fft_test_abs = np.abs(fft_test_sh)*highpass_filter

    # Log-polar
    R = fft_pattern.shape[0] // 2.0
    M = 2 * R / np.log(R)
    pattern_polar = cv2.logPolar(src=fft_pattern_abs,
                                 center=(R, R),
                                 M=M,
                                 flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    test_polar = cv2.logPolar(src=fft_test_abs,
                              center=(R, R),
                              M=M,
                              flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

    # 2D FFT
    fft_pattern_polar = np.fft.fft2(pattern_polar)
    fft_test_polar = np.fft.fft2(test_polar)

    # Phase correlation
    R = np.conj(fft_pattern_polar) * fft_test_polar
    R = R / np.abs(R)
    corr_norm = np.fft.ifft2(R)
    corr_norm_abs = np.abs(corr_norm)

    # Get rotation and scale
    wsp_kata, wsp_logr = np.unravel_index(
        np.argmax(corr_norm_abs), corr_norm_abs.shape)

    print('wsp_kata = {}, wsp_logr = {}'.format(wsp_kata, wsp_logr))

    # rozmiar_logr x rozmiar_kata to rozmiar obrazu logpolar
    rozmiar_logr = test_polar.shape[1]
    rozmiar_kata = test_polar.shape[0]
    if wsp_logr > rozmiar_logr // 2:
        wykl = rozmiar_logr - wsp_logr  # powiekszenie
    else:
        wykl = - wsp_logr  # pomniejszenie

    # gdzie A = (wsp_kata * 360.0 ) /rozmiar_kata
    A = (wsp_kata * 360.0) / rozmiar_kata
    # gdzie M to parametr funkcji cv2.logPolar, a wykl wyliczamy jako:
    scale = np.exp(wykl / M)
    kat1 = -A
    kat2 = 180 - A

    print('Kat1 = {}, kat2 = {}'.format(kat1, kat2))

    offset = int((test_img.shape[0]-pattern_img.shape[0])//2.0)
    im = np.zeros(test_img.shape)
    im[offset:pattern_img.shape[0]+offset,
       offset:pattern_img.shape[1]+offset] = pattern_img

    srodekTrans = (im.shape[0] // 2, im.shape[1] // 2)
    macierz_translacji1 = cv2.getRotationMatrix2D(
        (srodekTrans[0], srodekTrans[1]), kat1, scale)
    macierz_translacji2 = cv2.getRotationMatrix2D(
        (srodekTrans[0], srodekTrans[1]), kat2, scale)

    obraz_obrocony_przeskalowany1 = cv2.warpAffine(
        im, macierz_translacji1, pattern.shape)
    obraz_obrocony_przeskalowany2 = cv2.warpAffine(
        im, macierz_translacji2, pattern.shape)

    # 2D FFT
    obraz_obrocony_przeskalowany_fft1 = np.fft.fft2(
        obraz_obrocony_przeskalowany1)
    obraz_obrocony_przeskalowany_fft2 = np.fft.fft2(
        obraz_obrocony_przeskalowany2)

    # Phase correlation
    R1 = np.conj(obraz_obrocony_przeskalowany_fft1)*fft_test
    R1 = R1 / np.abs(R1)
    corr_norm1 = np.fft.ifft2(R1)
    corr_norm_abs1 = np.abs(corr_norm1)

    R2 = np.conj(obraz_obrocony_przeskalowany_fft2)*fft_test
    R2 = R2 / np.abs(R2)
    corr_norm2 = np.fft.ifft2(R2)
    corr_norm_abs2 = np.abs(corr_norm2)

    y1, x1 = np.unravel_index(
        np.argmax(corr_norm_abs1), corr_norm_abs1.shape)
    y2, x2 = np.unravel_index(
        np.argmax(corr_norm_abs2), corr_norm_abs2.shape)

    if np.max(corr_norm_abs1) > np.max(corr_norm_abs2):
        x_best = x1
        y_best = y1
        kat_best = kat1
        pattern_best = obraz_obrocony_przeskalowany1
        corr_best = corr_norm1
        macierz_translacji_best = macierz_translacji1
    else:
        x_best = x2
        y_best = y2
        kat_best = kat2
        pattern_best = obraz_obrocony_przeskalowany2
        corr_best = corr_norm2
        macierz_translacji_best = macierz_translacji2

    print('(x_best, y_best) = ({}, {}), kat_best = {}'.format(
        x_best, y_best, kat_best))

    M = cv2.getRotationMatrix2D((test_img.shape[0]//2,
                                 test_img.shape[1]//2), -kat_best, 1)
    test_img = cv2.warpAffine(test_img, M,
                              test_img.shape)

    if abs(kat_best) < 1:
        y_best = y_best - 128
        M = np.float32([[1, 0, -x_best], [0, 1, -y_best]])
        test_img = cv2.warpAffine(test_img, M,
                                  test_img.shape)

    # plt.figure()
    # plt.imshow(pattern_img_copy, cmap='gray')
    # plt.title('Pattern')

    plt.figure()
    plt.imshow(test_img, cmap='gray')
    plt.title('Transformed, angle = {}'.format(i))

plt.show()
