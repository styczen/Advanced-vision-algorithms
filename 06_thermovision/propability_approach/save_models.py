import cv2
import numpy as np
import sys
import os

THRESHOLD = 40
MIN_AREA = 2000

DIR = os.path.dirname(sys.argv[0])
# DIR = os.path.dirname(os.path.realpath(__file__))
cap = cv2.VideoCapture(DIR + '/../vid1_IR.avi')


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
iPedestrian = 26

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

            for i in range(updated_stats.shape[0]):
                if updated_stats[i, 4] > MIN_AREA:
                    cv2.rectangle(img=frame,
                                  pt1=(updated_stats[i, 0], updated_stats[i, 1]),
                                  pt2=(updated_stats[i, 0] + updated_stats[i, 2], updated_stats[i, 1] + updated_stats[i, 3]),
                                  color=(0, 0, 255),
                                  thickness=2)

        mask_filtered = cv2.cvtColor(mask_filtered, cv2.COLOR_GRAY2BGR)

        combined_imgs = np.hstack([mask_filtered, frame])
        cv2.imshow('Combined', combined_imgs)

        key = cv2.waitKey(0)
        if key & 0xff == ord('q'):
            break
        elif key & 0xff == ord('s'):
            for i in range(updated_stats.shape[0]):
                if updated_stats[i, 4] > MIN_AREA:
                    ROI = gray[stats[i, 1]:stats[i, 1]+stats[i, 3], stats[i, 0]:stats[i, 0]+stats[i, 2]]
                    cv2.imwrite(DIR + '/samples/sample_%06d.png' % iPedestrian, ROI)
                    iPedestrian += 1
                    print('Samples saved: {}'.format(iPedestrian))
    else:
        break

cv2.destroyAllWindows()
cap.release()
