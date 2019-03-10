import cv2
import numpy as np
import matplotlib.pyplot as plt

TP = 0
TN = 0
FP = 0
FN = 0

# - THRESHOLDs:
#   - pedestrains = 10
#   - highway = 9
#   - office = 1 (not a lot of movement, so subtracting two frames does not give good results)

THRESHOLD = 9

# Uncomment proper path to 'Sekwencje_testowe' directory
# DIR = '/home/student/dodatkowe/zaw2019/bstyczen/01_movement_detection/Sekwencje_testowe/'

# DIR = '/home/bartek/learning/Advanced-vision-algorithms/01_movement_detection/Sekwencje_testowe/pedestrians/'
DIR = '/home/bartek/learning/Advanced-vision-algorithms/01_movement_detection/Sekwencje_testowe/highway/'
# DIR = '/home/bartek/learning/Advanced-vision-algorithms/01_movement_detection/Sekwencje_testowe/office/'

f = open(DIR + '/temporalROI.txt', 'r')
line = f.readline()
roi_start, roi_end = line.split()
roi_start = int(roi_start)
roi_end = int(roi_end)

prev_frame = cv2.imread(DIR+'/input/in000' + str(roi_start) + '.jpg', cv2.IMREAD_GRAYSCALE)

for i in range(roi_start+1, roi_end, 1):
    fig = plt.figure(1)

    current_img_color = cv2.imread(DIR+'/input/in%06d.jpg' % i)
    current_img = current_img_color.copy()

    current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(current_img, prev_frame)
    cv2.medianBlur(diff, 5, diff)

    cv2.imshow('Diff', diff)

    mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)

    cv2.imshow('Mask', mask)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    cv2.imshow('Morphed mask', mask)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    cv2.imshow("Labels", np.uint8(labels / stats.shape[0] * 255))

    if stats.shape[0] > 1:      # czy sa jakies obiekty
        tab = stats[1:,4]       # wyciecie 4 kolumny bez pierwszego elementu
        pi = np.argmax( tab )   # znalezienie indeksu najwiekszego elementu
        pi = pi + 1             # inkrementacja bo chcemy indeks w stats, a nie w tab

        # wyrysownie bbox
        cv2.rectangle(img=current_img_color,
                      pt1=(stats[pi, 0], stats[pi, 1]),
                      pt2=(stats[pi, 0]+stats[pi, 2], stats[pi, 1]+stats[pi, 3]),
                      color=(0, 0, 255),
                      thickness=2)
        cv2.circle(img=current_img_color,
                   center=(np.int(centroids[pi, 0]), np.int(centroids[pi, 1])),
                   radius=2,
                   color=(0, 255, 0),
                   thickness=3)

        # wypisanie informacji o polu i numerze najwiekszego elementu
        cv2.putText(img=current_img_color,
                    text="%d" % stats[pi, 4],
                    org=(stats[pi, 0], stats[pi, 1]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0))
        cv2.putText(img=current_img_color,
                    text="%d" % pi,
                    org=(np.int(centroids[pi, 0]), np.int(centroids[pi, 1])),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0,255,0))

    cv2.imshow('Labeled frame', current_img_color)

    # Evaluate
    gt_img = cv2.imread(DIR+'/groundtruth/gt%06d.png' % i)
    cv2.imshow('gt', gt_img)

    TP_M = np.logical_and((mask == 255), (gt_img[:, :, 0] == 255))
    TP_S = np.sum(TP_M)

    FP_M = np.logical_and((mask == 255), (gt_img[:, :, 0] == 0))
    FP_S = np.sum(FP_M)

    FN_M = np.logical_and((mask == 0), (gt_img[:, :, 0] == 255))
    FN_S = np.sum(FN_M)

    TP = TP + TP_S
    FP = FP + FP_S
    FN = FN + FN_S

    if cv2.waitKey(1) == 27:
        break

    prev_frame = current_img

P = TP / (TP + FP)
R = TP / (TP + FN)
F1 = 2*P*R/(P + R)

print('P:', P)
print('R:', R)
print('F1:', F1)
