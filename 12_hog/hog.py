#!/usr/bin/python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import scipy.misc
import scipy.ndimage.filters
import math

DIR = os.path.dirname(sys.argv[0])


def gradient(img):
    """Calculates gradient and orientation."""
    gradRx = scipy.ndimage.filters.convolve1d(np.int32(img[:, :, 0]), np.array([-1, 0, 1]), 1)
    gradRy = scipy.ndimage.filters.convolve1d(np.int32(img[:, :, 0]), np.array([-1, 0, 1]), 0)

    gradGx = scipy.ndimage.filters.convolve1d(np.int32(img[:, :, 1]), np.array([-1, 0, 1]), 1)
    gradGy = scipy.ndimage.filters.convolve1d(np.int32(img[:, :, 1]), np.array([-1, 0, 1]), 0)

    gradBx = scipy.ndimage.filters.convolve1d(np.int32(img[:, :, 2]), np.array([-1, 0, 1]), 1)
    gradBy = scipy.ndimage.filters.convolve1d(np.int32(img[:, :, 2]), np.array([-1, 0, 1]), 0)

    # Gradient
    gradR = np.sqrt(gradRx ** 2 + gradRy ** 2)
    gradG = np.sqrt(gradGx ** 2 + gradGy ** 2)
    gradB = np.sqrt(gradBx ** 2 + gradBy ** 2)

    grad1 = gradB.copy()        # poczatkowo wynikowa macierz to gradient skladowej B
    m1 = gradB - gradG          # m1 - tablica pomocnicza do wyznaczania maksimum miedzy skladowymi B i G
    grad1[m1<0] = gradG[m1<0]   # w macierzy wynikowej gradienty skladowej B sa podmieniane na wieksze od nich gradienty skladowej G

    grad2 = gradR.copy()
    m2 = gradR - grad1
    grad2[m2<0] = grad1[m2<0]   # max

    # Orientation
    orientationR = np.rad2deg(np.arctan2(gradRy, gradRx))
    orientationG = np.rad2deg(np.arctan2(gradGy, gradGx))
    orientationB = np.rad2deg(np.arctan2(gradBy, gradBx))

    orie1 = orientationG.copy()
    orie1[m1<0] = orientationG[m1<0]

    orie2 = orientationR.copy()
    orie2[m2<0] = orie1[m2<0]

    grad2[0,  0:-1] = 0
    grad2[-1, 0:-1] = 0
    grad2[0:-1, 0] = 0
    grad2[0:-1, -1] = 0

    orie2[0,  0:-1] = 0
    orie2[-1, 0:-1] = 0
    orie2[0:-1, 0] = 0
    orie2[0:-1, -1] = 0

    return grad2, orie2


def histogram(SXY, DIR, XX, YY):
    """Calculates histogram of oriented gradients."""
    # Obliczenia histogramow
    cellSize = 8  # rozmiar komorki
    YY_cell = np.int32(YY/cellSize)
    XX_cell = np.int32(XX/cellSize)

    # Kontener na histogramy - zakladamy, ze jest 9 przedzialow
    hist = np.zeros([YY_cell, XX_cell, 9], np.float32)
    # Iteracja po komorkach na obrazie
    for jj in range(0, YY_cell):
        for ii in range(0, XX_cell):
            # Wyciecie komorki
            M = SXY[jj*cellSize:(jj+1)*cellSize, ii*cellSize:(ii+1)*cellSize]
            T = DIR[jj*cellSize:(jj+1)*cellSize, ii*cellSize:(ii+1)*cellSize]
            M = M.flatten()
            T = T.flatten()

            # Obliczenie histogramu
            for k in range(0, cellSize*cellSize):
                m = M[k]
                t = T[k]

                # Usuniecie ujemnych kata (zalozenie katy w stopniach)
                if t < 0:
                    t = t + 180

                # Wyliczenie przedzialu
                t0 = np.floor((t - 10) / 20) * 20 + 10  # Przedzial ma rozmiar 20, srodek to 20

                # Przypadek szczegolny tj. t0 ujemny
                if t0 < 0:
                    t0 = 170  # Wyznaczenie indeksow przedzialu

                i0 = int((t0-10)/20)
                i1 = i0+1

                # Zawijanie
                if i1 == 9:
                    i1 = 0

                # Obliczenie odleglosci do srodka przedzialu
                d = min(abs(t-t0), 180 - abs(t-t0)) / 20

                # print('i0 = {}, i1 = {}'.format(i0, i1))

                # Aktualizacja histogramu
                hist[jj, ii, i0] = hist[jj, ii, i0] + m*(1-d)
                hist[jj, ii, i1] = hist[jj, ii, i1] + m*(d)
    return hist


def hist_normalize(hist, XX, YY):
    """Normalize calculated histograms."""
    # Normalizacja w blokach
    cellSize = 8  # rozmiar komorki
    YY_cell = np.int32(YY/cellSize)
    XX_cell = np.int32(XX/cellSize)

    e = math.pow(0.00001, 2)
    F = []
    for jj in range(0, YY_cell-1):
        for ii in range(0, XX_cell-1):
            H0 =  hist[jj,ii,:]
            H1 =  hist[jj,ii+1,:]
            H2 =  hist[jj+1,ii,:]
            H3 =  hist[jj+1,ii+1,:]
            H = np.concatenate((H0, H1, H2, H3))
            n = np.linalg.norm(H)
            Hn = H/np.sqrt(math.pow(n,2)+e)
            F = np.concatenate((F,Hn))
    return F


def HOGpicture(w, bs=8): # w - histogramy gradientow obrazu, bs - rozmiar komorki (u nas 8)
    bim1 = np.zeros((bs, bs))
    bim1[np.round(bs//2):np.round(bs//2)+1, :] = 1
    bim = np.zeros(bim1.shape+(9,))
    bim[:, :, 0] = bim1
    for i in range(0, 9):  # 2:9
        bim[:, :, i] = scipy.misc.imrotate(bim1, -i*20,'nearest')/255
    Y, X, Z = w.shape
    w[w < 0] = 0
    im = np.zeros((bs*Y, bs*X))
    for i in range(Y):
        iisl = (i)*bs
        iisu = (i+1)*bs
        for j in range(X):
            jjsl = j*bs
            jjsu = (j+1)*bs
            for k in range(9):
                im[iisl:iisu, jjsl:jjsu] += bim[:,:,k]*w[i,j,k]
    return im


im = cv2.imread(DIR + '/pedestrians/' + 'pos/' + 'per00060.ppm')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

gradient, orientation = gradient(im)

hist = histogram(gradient, orientation, im.shape[1], im.shape[0])

hist_norm = hist_normalize(hist, im.shape[1], im.shape[0])

temp = HOGpicture(hist)

plt.figure()
plt.imshow(gradient, cmap='gray')
plt.title('Gradient')

plt.figure()
plt.imshow(temp, cmap='gray')
plt.title('Histograms')

plt.figure()
plt.imshow(orientation, cmap='gray')
plt.title('Orientation')

plt.show()
