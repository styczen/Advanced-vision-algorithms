#!/usr/bin/env python3
import cv2
import matplotlib
import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn import svm

from hog_descriptor import hog
%matplotlib auto

def predict(img, clf):
    hist = hog(img)
    hist = hist.reshape(1, -1)
    p = clf.predict(hist)
    return p

# %%

# Calculate HOG descriptor
HOG_data = np.zeros([2*100, 3781], np.float32)
for i in range(0, 100):
    print('Image {}'.format(i+1))
    IP = cv2.imread('pedestrians/pos/per%05d.ppm' % (i+1))
    IN = cv2.imread('pedestrians/neg/neg%05d.png' % (i+1))
    F = hog(IP)
    HOG_data[i, 0] = 1
    HOG_data[i, 1:] = F
    F = hog(IN)
    HOG_data[i+100, 0] = 0
    HOG_data[i+100, 1:] = F
    
# %%

# Train classifier
labels = HOG_data[:, 0]
data = HOG_data[:, 1:]

clf = svm.SVC(kernel='linear',
              C=1.0)

clf.fit(data, labels)

# %%

# Test classifier on training data
lp = clf.predict(data)

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(labels.shape[0]):
    if lp[i] == 1 and labels[i] == 1:
        TP += 1
    elif lp[i] == 0 and labels[i] == 0:
        TN += 1
    elif lp[i] == 1 and labels[i] == 0:
        FP += 1
    elif lp[i] == 0 and labels[i] == 1:
        FN += 1

acc = (TP + TN) / (2*100)
print('Accuracy = {}%'.format(acc*100))

# %%

# Validate on another images
nr_validate_imgs = 825
HOG_data = np.zeros([2*nr_validate_imgs, 3781], np.float32)
for i in range(100, 100+nr_validate_imgs):
    print('Image {}'.format(i+1))
    IP = cv2.imread('pedestrians/pos/per%05d.ppm' % i)
    IN = cv2.imread('pedestrians/neg/neg%05d.png' % i)
    F = hog(IP)
    HOG_data[i-100, 0] = 1
    HOG_data[i-100, 1:] = F
    F = hog(IN)
    HOG_data[(i-100)+100, 0] = 0
    HOG_data[(i-100)+100, 1:] = F

# %%

# Predict on validation data
labels = HOG_data[:, 0]
data = HOG_data[:, 1:]

lp = clf.predict(data)

TP = 0
TN = 0
FP = 0
FN = 0

for i in range(labels.shape[0]):
    if lp[i] == 1 and labels[i] == 1:
        TP += 1
    elif lp[i] == 0 and labels[i] == 0:
        TN += 1
    elif lp[i] == 1 and labels[i] == 0:
        print('FP, image number = {}'.format(i+100))
        FP += 1
    elif lp[i] == 0 and labels[i] == 1:
        print('FN, image number = {}'.format(i+100))
        FN += 1

acc = (TP + TN) / (2*nr_validate_imgs)
print('Accuracy on validation data = {:.2f}%'.format(acc*100))

# %%

# Test classifier on real images
window_size = (64, 128)
img_nr = 4

# Scale values chosen experimentally
scale = [-1, 0.7, 0.5, 1.0, 1.0]

img = cv2.imread('testImages/testImage{}.png'.format(img_nr))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_org = img.copy()

img = cv2.resize(src=img,
                 dsize=(0, 0),
                 fx=scale[img_nr],
                 fy=scale[img_nr])
img_result = img.copy()

# %%

# Test on full image
stride = 8
for row in range(0, img.shape[0]-window_size[1], stride):
    for col in range(0, img.shape[1]-window_size[0], stride):
        if predict(img[row:row+window_size[1], 
                       col:col+window_size[0], :], clf):
            pt1 = (int(col/scale[img_nr]), int(row/scale[img_nr]))
            pt2 = (int(pt1[0]+window_size[0]/scale[img_nr]), 
                   int(pt1[1]+window_size[1]/scale[img_nr]))
            img_org = cv2.rectangle(img_org, 
                                    pt1=pt1,
                                    pt2=pt2,
                                    color=(255, 0, 0))

plt.imshow(img_org)
plt.show()

# %%

# Save result to file
matplotlib.image.imsave('testImages/testImage{}_result.png'.format(img_nr), 
                        img_org)

# %%

# Multiscale approach
img_nr = 4
window_size = (64, 128)
stride = 8
img = cv2.imread('testImages/testImage{}.png'.format(img_nr))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_org = img.copy()

for i in range(9, 0, -1):
    scale = 0.1 * i
    print('Scale = {}'.format(scale))
    img_curr_scale = cv2.resize(src=img,
                                dsize=(0, 0),
                                fx=scale,
                                fy=scale)

    for row in range(0, img_curr_scale.shape[0]-window_size[1], stride):
        for col in range(0, img_curr_scale.shape[1]-window_size[0], stride):
            if predict(img_curr_scale[row:row+window_size[1], 
                                      col:col+window_size[0], :], clf):
    
                pt1 = (int(col/scale), int(row/scale))
                pt2 = (int(pt1[0]+window_size[0]/scale), 
                       int(pt1[1]+window_size[1]/scale))
                img_org = cv2.rectangle(img_org, 
                                        pt1=pt1,
                                        pt2=pt2,
                                        color=(255, 0, 0))

plt.imshow(img_org)
plt.show()

# %%

# Save result to file
matplotlib.image.imsave('testImages/testImage{}_multiscale.png'.format(img_nr), 
                        img_org)
