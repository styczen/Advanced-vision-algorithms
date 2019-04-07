import cv2
import numpy as np
import os

DIR = os.getcwd()
THRESHOLD = 40


def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < 50:
                return True
            elif i == row1 - 1 and j == row2 - 1:
                return False


roi_mask = np.vstack((np.zeros((95, 480), dtype=np.uint8), np.ones((360-95, 480), dtype=np.uint8)))
cap = cv2.VideoCapture(DIR + '/vid1_IR.avi')

while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('IR', gray)

        gray = gray * roi_mask
        retval, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
        # cv2.imshow('Mask', mask)

        mask_filtered = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask_filtered = cv2.dilate(mask_filtered, kernel, iterations=1)
        cv2.imshow('Mask filtered', mask_filtered)

        contours, hier = cv2.findContours(mask_filtered, cv2.RETR_EXTERNAL, 2)

        LENGTH = len(contours)
        if LENGTH > 0:
            status = np.zeros((LENGTH, 1))

            for i, cnt1 in enumerate(contours):
                x = i
                if i != LENGTH - 1:
                    for j, cnt2 in enumerate(contours[i + 1:]):
                        x = x + 1
                        dist = find_if_close(cnt1, cnt2)
                        if dist == True:
                            val = min(status[i], status[x])
                            status[x] = status[i] = val
                        else:
                            if status[x] == status[i]:
                                status[x] = i + 1

            unified = []
            maximum = int(status.max()) + 1
            for i in range(maximum):
                pos = np.where(status == i)[0]
                if pos.size != 0:
                    contours_l = [contours[i] for i in pos]
                    cont = np.vstack(contours_l)
                    hull = cv2.convexHull(cont)
                    unified.append(hull)

            cv2.drawContours(frame, unified, -1, (0, 255, 0), 2)
            cv2.drawContours(mask_filtered, unified, -1, 255, -1)

            cv2.imshow('Frame', frame)
            cv2.imshow('Mask', mask_filtered)
            if cv2.waitKey(1) == 27:
                break

cv2.destroyAllWindows()
