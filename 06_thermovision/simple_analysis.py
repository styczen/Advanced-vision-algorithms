import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

DIR = os.getcwd()
THRESHOLD = 40
MIN_AREA = 500

cap = cv2.VideoCapture(DIR + '/vid1_IR.avi')


def connectBlobs(stats, centroids):
    updated_stats = []
    for i in range(1, stats.shape[0]):
        if stats[i, 4] > MIN_AREA and stats[i, 2] < stats[i, 3]:
            for j in range(1, stats.shape[0]):
                if i == j:
                    continue
                if centroids[i, 0] - 30 < centroids[j, 0] and centroids[j, 0] < centroids[i, 0] + 30:
                    left = stats[i, 0] if stats[i, 0] < stats[j, 0] else stats[j, 0]
                    top = stats[i, 1] if stats[i, 1] < stats[j, 1] else stats[j, 1]
                    width = stats[i, 2] if stats[i, 2] > stats[j, 2] else stats[j, 2]
                    new_stats = [left, top, width, stats[i, 3]+stats[j, 3]]
                    updated_stats.append(new_stats)
    return np.array(updated_stats)


while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_temp = frame.copy()
        # cv2.imshow('IR', gray)

        retval, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
        # cv2.imshow('Mask', mask)

        mask_filtered = cv2.medianBlur(mask, 5)
        # cv2.imshow('Mask filtered', mask_filtered)

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filtered)
        # cv2.imshow('Labels', np.uint8(labels / stats.shape[0] * 255))

        if retval > 0:  # whether there are some objects labeled
            updated_stats = connectBlobs(stats, centroids)
            print(updated_stats.shape)
            for i in range(updated_stats.shape[0]):
                cv2.rectangle(img=frame_temp,
                              pt1=(updated_stats[i, 0], updated_stats[i, 1]),
                              pt2=(updated_stats[i, 0] + updated_stats[i, 2], updated_stats[i, 1] + updated_stats[i, 3]),
                              color=(0, 0, 255),
                              thickness=2)

            for i in range(1, stats.shape[0]):
                if stats[i, 4] > MIN_AREA and stats[i, 2] < stats[i, 3]:
                    print(centroids[i])
                    cv2.rectangle(img=frame,
                                  pt1=(stats[i, 0], stats[i, 1]),
                                  pt2=(stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]),
                                  color=(0, 0, 255),
                                  thickness=2)

                    cv2.circle(img=frame,
                               center=(np.int(centroids[i, 0]), np.int(centroids[i, 1])),
                               radius=2,
                               color=(0, 255, 0),
                               thickness=3)

        cv2.imshow('Frame', frame)
        cv2.imshow('Frame connected', frame_temp)

        if cv2.waitKey(0) & 0xff == ord('q'):
            break
    else:
        break

cap.release()
