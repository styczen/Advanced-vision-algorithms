#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import os
import math
import numpy as np

DIR = os.getcwd()


kernel_size = 55 # rozmiar rozkladu
mouseX, mouseY = (830, 430) # przykladowe wspolrzedne

def track_init(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.rectangle(param, (x-kernel_size//2, y- kernel_size//2),
                      (x + kernel_size//2, y + kernel_size//2), (0, 255, 0), 2)
        mouseX, mouseY = x, y # Wczytanie pierwszego obrazka


I = cv2.imread('track_seq/track00100.png')
cv2.namedWindow('Tracking')
cv2.setMouseCallback('Tracking', track_init, param=I)

# Pobranie klawisza
while True:
    cv2.imshow('Tracking', I)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:   # ESC
        break

# gaussian mask
sigma = kernel_size / 6
x = np.arange(0, kernel_size, 1, float)
y = x[:, np.newaxis]
x0 = y0 = kernel_size // 2
G = 1/(2*math.pi*sigma**2)*np.exp(-0.5*((x-x0)**2 + (y-y0)**2) / sigma**2)

xS = mouseX - kernel_size//2
yS = mouseY - kernel_size//2
I_HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)

I_H = I_HSV[:, :, 0]
hist_q = np.zeros((256, 1), float)
for u in range(256):
    mask = I_H[yS:yS+kernel_size, xS:xS+kernel_size] == u
    hist_q[u] = np.sum(G[mask])
hist_q = hist_q / hist_q.max()

for i in range(101, 201):
    I = cv2.imread('track_seq/track%05d.png' % i)
    I_HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    I_H = I_HSV[:, :, 0]
    hist_p = np.zeros((256, 1), float)
    for u in range(256):
        mask = I_H[yS:yS + kernel_size, xS:xS + kernel_size] == u
        hist_p[u] = np.sum(G[mask])
    hist_p = hist_p / hist_p.max()

    ro = np.sqrt(hist_p * hist_q)

    window = np.zeros((kernel_size, kernel_size))
    for row in range(kernel_size):
        for col in range(kernel_size):
            window[row, col] = G[row, col] * ro[I_H[row, col]]

    moments = cv2.moments(window)
    if moments['m00'] < 0.00001:
        continue
    xc = int(moments['m10'] / moments['m00'])
    yc = int(moments['m01'] / moments['m00'])

    xS = xS + (yc - kernel_size//2)
    yS = yS + (xc - kernel_size//2)

    cv2.rectangle(I, (xS, yS), (xS + kernel_size, yS + kernel_size), (0, 255, 0), 2)

    cv2.imshow('here', I)
    k = cv2.waitKey(0) & 0xFF
    if k == 27:   # ESC
        break

cv2.destroyAllWindows()
