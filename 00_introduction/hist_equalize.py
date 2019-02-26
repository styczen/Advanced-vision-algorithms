import cv2

image = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)

# Global histogram equalization
image_global = cv2.equalizeHist(image)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# clipLimit - maksymalna wysokosc slupka histogramu - wartosci powyzej rozdzielana sa pomiedzy sasiadow
# tileGridSize - rozmiar pojedycznczego bloku obrazu (metoda lokalna, dzialana rozdzielnych blokach obrazu)
image_clahe = clahe.apply(image)

cv2.imshow('Original', image)
cv2.imshow('Global', image_global)
cv2.imshow('CLAHE', image_clahe)

cv2.waitKey(0)
cv2.destroyAllWindows()