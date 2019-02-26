import matplotlib
import matplotlib.pyplot as plt


def rgb2gray(img):
    return 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]


# Loading image
image = plt.imread('mandril.jpg')

# RGB -> Grayscale
grayscale = rgb2gray(image)
plt.figure(1)
plt.gray()
plt.imshow(grayscale)
plt.title('Mandril - grayscale')
plt.axis('off')

# RGB -> HSV
I_HSV = matplotlib.colors.rgb_to_hsv(image)
plt.figure(2)
plt.imshow(I_HSV)
plt.title('Mandril - HSV')

plt.show()

