import cv2
import numpy as np
import sys
import os

THRESHOLD = 40

DIR = os.path.dirname(sys.argv[0])

nr_samples = 38

DPM = np.zeros((192, 64), np.uint8)

for i in range(nr_samples+1):
    sample = cv2.imread(DIR + '/samples/sample_%06d.png' % i, cv2.IMREAD_GRAYSCALE)

    sample = cv2.resize(sample, (64, 192), interpolation=cv2.INTER_LINEAR)

    retval, mask = cv2.threshold(sample, THRESHOLD, 1, cv2.THRESH_BINARY)

    mask_filtered = cv2.medianBlur(mask, 5)
    # kernel = np.ones((5, 5), np.uint8)
    # mask_filtered = cv2.dilate(mask_filtered, kernel, iterations=1)

    DPM += mask_filtered

    # combined = np.hstack([sample, mask_filtered*255, DPM])
    # cv2.imshow('Combined', combined)

    # cv2.imshow('sample', sample)
    # cv2.imshow('mask', mask_filtered)
    # cv2.imshow('dpm', DPM)
    #
    # if cv2.waitKey(0) & 0xff == ord('q'):
    #     break

cv2.imwrite(DIR + '/dpm.png', DPM)
print('...done')
cv2.destroyAllWindows()
