import cv2
import numpy as np

TP = 0
TN = 0
FP = 0
FN = 0

THRESHOLD = 10

DIR = '/home/student/dodatkowe/zaw2019/bstyczen/01_movement_detection/Sekwencje_testowe/'

prev_frame = cv2.imread(DIR+'/pedestrians/input/in000306.jpg', cv2.IMREAD_GRAYSCALE)

for i in range(307, 1100, 1):
    current_img_color = cv2.imread(DIR+'/pedestrians/input/in%06d.jpg' % i)
    current_img = current_img_color.copy()

    current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(current_img, prev_frame)
    cv2.imshow('Diff', diff)

    mask = cv2.threshold(diff, THRESHOLD, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.medianBlur(mask, 5)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    cv2.imshow('Mask', mask)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    cv2.imshow("Labels", np.uint8(labels / stats.shape[0] * 255))

    if stats.shape[0] > 1:      # czy sa jakies obiekty
        tab = stats[1:,4]       # wyciecie 4 kolumny bez pierwszego elementu
        pi = np.argmax( tab )   # znalezienie indeksu najwiekszego elementu
        pi = pi + 1             # inkrementacja bo chcemy indeks w stats, a nie w tab

        # wyrysownie bbox
        cv2.rectangle(current_img_color,
                      (stats[pi,0], stats[pi,1]), (stats[pi,0]+stats[pi,2], stats[pi,1]+stats[pi,3]),
                      (0,0,255),
                      2)
        cv2.circle(current_img_color,
                   (np.int(centroids[pi,0]), np.int(centroids[pi,1])),
                   2,
                   (0,255,0),
                   3)

        # wypisanie informacji o polu i numerze najwiekszego elementu
        cv2.putText(current_img_color, "%d" % stats[pi,4],
                    (stats[pi,0],stats[pi,1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,0,0))
        cv2.putText(current_img_color, "%d" % pi,
                    (np.int(centroids[pi,0]), np.int(centroids[pi,1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0))
    cv2.imshow('Labeled frame', current_img_color)

    # Evaluate
    gt_img = cv2.imread(DIR+'/pedestrians/groundtruth/gt%06d.png' % i)
    cv2.imshow('gt', gt_img)

    TP_M = np.logical_and((mask==255), (gt_img[:,:,0]==255))
    TP_S = np.sum(TP_M)

    FP_M = np.logical_and((mask==255), (gt_img[:,:,0]==0))
    FP_S = np.sum(FP_M)

    FN_M = np.logical_and((mask==0), (gt_img[:,:,0]==255))
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
