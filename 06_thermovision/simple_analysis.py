import cv2
import numpy as np
import os

DIR = os.getcwd()
THRESHOLD = 40
MIN_AREA = 2000

cap = cv2.VideoCapture(DIR + '/vid1_IR.avi')


def connect_blobs1(stats , centroids):
    for i in range(stats.shape[0]):
        # parent_blob_left, parent_blob_top = stats[i, 0], stats[i, 1]
        # parent_blob_right, parent_blob_bottom = parent_blob_left+stats[i, 2], parent_blob_top+stats[i, 3]
        # parent_x, parent_y = centroids[i, 0], centroids[i, 1]

        # parent_blob_area = stats[i, 4]
        # if parent_blob_area > MIN_AREA:
        for j in range(stats.shape[0]):
            parent_blob_left, parent_blob_top = stats[i, 0], stats[i, 1]
            parent_blob_right, parent_blob_bottom = parent_blob_left + stats[i, 2], parent_blob_top + stats[i, 3]
            parent_blob_area = stats[i, 4]
            parent_x, parent_y = centroids[i, 0], centroids[i, 1]

            left, top = stats[j, 0], stats[j, 1]
            right, bottom = left+stats[j, 2], top+stats[j, 3]
            area = stats[j, 4]
            x, y = centroids[j, 0], centroids[j, 1]
            if i != j and parent_x - 30 < x < parent_x + 30:
                new_left = parent_blob_left if parent_blob_left < left else left
                new_top = parent_blob_top if parent_blob_top < top else top
                new_right = parent_blob_right if parent_blob_right > right else right
                new_bottom = parent_blob_bottom if parent_blob_bottom > bottom else bottom
                stats[i, :] = np.array([new_left, new_top, new_right-new_left, new_bottom-new_top, parent_blob_area+area])


    for i in range(stats.shape[0]):
        for j in range(stats.shape[0]):
            if i != j and np.all(stats[i, :4] == stats[j, :4]):
                stats[j, 4] = stats[i, 4]

    if stats.shape[0] > 0:
        stats = np.unique(stats, axis=0)

    return stats


roi_mask = np.vstack((np.zeros((100, 480), dtype=np.uint8), np.ones((360-100, 480), dtype=np.uint8)))

while cap.isOpened():
    ret, frame = cap.read()
    if ret is True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_temp = frame.copy()
        # cv2.imshow('IR', gray)

        gray = gray * roi_mask
        retval, mask = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
        # cv2.imshow('Mask', mask)

        mask_filtered = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask_filtered = cv2.dilate(mask_filtered, kernel, iterations=2)
        # cv2.imshow('Mask filtered', mask_filtered)

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_filtered)
        # cv2.imshow('Labels', np.uint8(labels / stats.shape[0] * 255))

        if retval > 0:  # whether there are some objects labeled

            # Extract all objects except whole image center of mass and bbox coords
            stats = stats[1:, :]
            centroids = centroids[1:, :]

            stats_copy = stats.copy()

            updated_stats = connect_blobs1(stats_copy, centroids)
            # print('*********************')
            # print('Stats')
            # print(stats)
            # print('---------------------')
            # print('Updated stats')
            # print(updated_stats)
            # print('*********************')

            for i in range(updated_stats.shape[0]):
                if updated_stats[i, 4] > MIN_AREA:
                    cv2.rectangle(img=frame_temp,
                                  pt1=(updated_stats[i, 0], updated_stats[i, 1]),
                                  pt2=(updated_stats[i, 0] + updated_stats[i, 2], updated_stats[i, 1] + updated_stats[i, 3]),
                                  color=(0, 0, 255),
                                  thickness=2)

            for i in range(stats.shape[0]):
                # if stats[i, 4] > MIN_AREA:
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

        mask_filtered = cv2.cvtColor(mask_filtered, cv2.COLOR_GRAY2BGR)

        # cv2.imshow('Mask filtered', mask_filtered)
        # cv2.imshow('Frame', frame)
        # cv2.imshow('Frame connected', frame_temp)

        combined_imgs = np.hstack([mask_filtered, frame, frame_temp])
        cv2.imshow('Combined', combined_imgs)

        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    else:
        break

cap.release()
