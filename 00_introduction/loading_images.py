import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

print('***********OpenCV part***********')
I = cv2.imread('mandril.jpg')
cv2.imshow("Mandril",I)             # wyswietlenie
cv2.waitKey(0)                      # oczekiwanie na klawisz
cv2.destroyAllWindows()             # zamkniecie wszystkich okien
cv2.imwrite("m.png",I)              # zapis obrazu do pliku

print("Shape:", I.shape)            # rozmiary /wiersze, kolumny, glebia/
print("Number of bytes:", I.size)   # liczba bajtow
print("Data type:", I.dtype)        # typ danych

print('***********Matplotlib part***********')

I = plt.imread('mandril.jpg')
plt.figure(1)                       # stworzenie figury
plt.imshow(I)                       # dodanie do niej obrazka
plt.title('Mandril')                # dodanie tytulu
plt.axis('off')                     # wylaczenie wyswietlania ukladu wspolrzednych
plt.show()                          # wyswietlnie calosci
plt.imsave('mandril.png', I)

x = [100, 150, 200, 250]
y = [50, 100, 150, 200]
plt.figure(2)                       # stworzenie figury
plt.plot(x, y, 'r.', markersize=10)
plt.show()                          # wyswietlnie calosci

print('***********Patches***********')
fig, ax = plt.subplots(1)
rect = Rectangle((50, 50), 50, 100, fill=False, ec='r') # ec - edge color
ax.add_patch(rect)
ax.plot()
plt.show()
