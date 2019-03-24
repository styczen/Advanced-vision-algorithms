import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time

DIR = os.getcwd()


def pyramid(im, max_scale):
    images = [im]
    for k in range(1, max_scale):
        images.append(cv2.resize(images[k-1], (0, 0), fx=0.5, fy=0.5))
    return images


def of(img1, img2, u0, v0, W=3, W2=1, dY=1, dX=1):
    u = np.zeros(img1.shape, dtype=np.int8)
    v = np.zeros(img1.shape, dtype=np.int8)

    for j in range(W-1, img1.shape[0]-(W-1)):
        for i in range(W-1, img1.shape[1]-(W-1)):
            img1_o = np.float32(img1[j - W2:j + W2 + 1, i - W2:i + W2 + 1])

            best_x, best_y = None, None
            min_dist = 100000.0
            for jo in range(j-W2, j+W2+1, dX):
                for io in range(i-W2, i+W2+1, dY):
                    jo_t = jo + u0[j, i]
                    io_t = io + v0[j, i]
                    img2_o = np.float32(img2[jo_t - W2:jo_t + W2 + 1, io_t - W2:io_t + W2 + 1])
                    dist = np.sqrt(np.sum(np.square(img1_o - img2_o)))
                    if dist < min_dist:
                        best_x, best_y = io-i, jo-j
                        min_dist = dist
            u[j, i] = best_y + u0[j, i]
            v[j, i] = best_x + v0[j, i]

    return u, v


if __name__ == '__main__':
    img1_color = cv2.imread(DIR + '/I.jpg')
    img2_color = cv2.imread(DIR + '/J.jpg')

    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    W = 3
    dX, dY = 1, 1
    W2 = int(np.floor(W / 2))

    nr_scales = 3
    pyramid_img1 = pyramid(img1, nr_scales)
    pyramid_img2 = pyramid(img2, nr_scales)

    # MULTISCALE
    u0 = np.zeros(pyramid_img1[-1].shape, dtype=np.int8)
    v0 = np.zeros(pyramid_img2[-1].shape, dtype=np.int8)
    u = None
    v = None

    ul = []
    vl = []

    print('Calculating multiscale optical flow...')
    start = time.process_time()

    for i in reversed(range(len(pyramid_img1))):
        print('Scale size: {}'.format(pyramid_img1[i].shape))

        u, v = of(pyramid_img1[i], pyramid_img2[i], u0, v0, W, W2, dY, dX)
        u0 = cv2.resize(u, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        v0 = cv2.resize(v, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

        ul.append(u)
        vl.append(v)

    stop = time.process_time()
    print('...done')

    print("Summary: W = {}, dX = {}, dY = {}, time = {:.2f} sec.".format(W, dX, dY, stop - start))

    # Show optical flow on every scale
    for i in range(len(ul)-1):
        plt.figure(i)
        plt.quiver(ul[i], vl[i])
        plt.gca().invert_yaxis()

    # Show result of multiscale optical flow
    plt.figure(len(ul)-1)
    plt.quiver(ul[-1], vl[-1])
    plt.title('Multiscale')
    plt.gca().invert_yaxis()

    # SINGLESCALE
    # Calculate optical flow without multiscale approach
    u0 = np.zeros(pyramid_img1[0].shape, dtype=np.int8)
    v0 = np.zeros(pyramid_img1[0].shape, dtype=np.int8)

    print('\nCalculating singlescale optical flow...')
    start = time.process_time()

    u_single_scale, v_single_scale = of(pyramid_img1[0], pyramid_img2[0], u0, v0, W, W2, dY, dX)

    stop = time.process_time()
    print('...done')

    print("Summary: W = {}, dX = {}, dY = {}, time = {:.2f} sec.".format(W, dX, dY, stop - start))

    plt.figure(len(pyramid_img1))
    plt.quiver(u_single_scale, v_single_scale)
    plt.title('Singlescale')
    plt.gca().invert_yaxis()

    print('\nComparison of multiscale and singlescale')
    print('u_m == u_s -> {}'.format(np.all(ul[-1] == u_single_scale)))
    print('v_m == v_s -> {}'.format(np.all(vl[-1] == v_single_scale)))

    plt.show()
