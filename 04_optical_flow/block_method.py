import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time


DIR = os.getcwd()


if __name__ == '__main__':
    img1_color = cv2.imread(DIR + '/I.jpg')
    img2_color = cv2.imread(DIR + '/J.jpg')

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(img1, img2)

    # img1 = cv2.medianBlur(img1, 3)
    # img2 = cv2.medianBlur(img2, 3)

    cv2.imshow('Frame 1', img1)
    cv2.imshow('Frame 2', img2)

    cv2.imshow('Diff', diff)

    W = 3
    dX, dY = 1, 1
    W2 = int(np.floor(W / 2))

    u = np.zeros(img1.shape)
    v = np.zeros(img2.shape)

    start = time.process_time()
    for j in range(W-1, img1.shape[0]-(W-1)):
        for i in range(W-1, img1.shape[1]-(W-1)):
            IO = np.float32(img1[j - W2:j + W2 + 1, i - W2:i + W2 + 1])

            best_x, best_y = -1, -1
            min_dist = 100000.0
            for jo in range(j-W2, j+W2+1, dX):
                for io in range(i-W2, i+W2+1, dY):
                    JO = np.float32(img2[jo - W2:jo + W2 + 1, io - W2:io + W2 + 1])
                    dist = np.sqrt(np.sum(np.square(JO - IO)))
                    if dist < min_dist:
                        best_x, best_y = io-i, jo-j
                        min_dist = dist
            u[j, i] = best_y
            v[j, i] = best_x
            
    stop = time.process_time()
    print("W = {}, dX = {}, dY = {}, time = {} sec.".format(W, dX, dY, stop-start))

    plt.figure(1)
    plt.quiver(u, v)
    plt.gca().invert_yaxis()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
