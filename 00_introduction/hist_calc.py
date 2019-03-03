import cv2
import matplotlib.pyplot as plt
import numpy as np

def hist(img):
    h = np.zeros((256, 1), np.float32)
    height, width = img.shape[:2]
    for y in range(height):
        for x in range(width):
            h[img[y,x]] = h[img[y,x]] + 1
    return h

if __name__ == '__main__':
    img = cv2.imread('lena.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    custom_hist = hist(img)
    opencv_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    
    print('sum(abs(custom_hist - opencv_hist)) = ', np.sum(np.abs(custom_hist - opencv_hist)))

    plt.figure(1)
    plt.plot(custom_hist)
    plt.title('Custom hist')
    
    plt.figure(2)
    plt.plot(opencv_hist)
    plt.title('OpenCV hist')

    plt.show()
    