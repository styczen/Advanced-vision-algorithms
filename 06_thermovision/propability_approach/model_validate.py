import cv2
import numpy as np
import sys
import os

THRESHOLD = 40
NR_SAMPLES = 39
PROB_THRESHOLD = 240
STRIDE = 1

DIR = os.path.dirname(sys.argv[0])


# Malisiewicz et al.
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

        # return only the bounding boxes that were picked
    return boxes[pick]


cap = cv2.VideoCapture(DIR + '/../vid1_IR.avi')

model = cv2.imread(DIR + '/dpm.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
model_h, model_w = model.shape[:2]

DPM_1 = model / NR_SAMPLES
DPM_0 = 1 - DPM_1

frame_bgr = cv2.imread(DIR + '/test_ir_frame.png')
h, w = frame_bgr.shape[:2]

nr_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame_all_bbox = frame.copy()
    # print(nr_frame)
    # nr_frame += 1
    # cv2.imshow('Frame', frame)
    # cv2.waitKey(0)
    # continue
    if ret is True:
        nr_frame += 1
        if nr_frame < 310:  # 60 - one person, 310 - two persons
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('IR', gray)

        retval, mask = cv2.threshold(gray, THRESHOLD, 1, cv2.THRESH_BINARY)
        # cv2.imshow('Mask', mask)

        result = np.zeros((360, 480), np.float32)
        best_x, best_y, max_prob = 0, 0, 0.0

        for i in range(0, h - model_h, STRIDE):
            for j in range(0, w - model_w, STRIDE):
                roi = mask[i:i + model_h, j:j + model_w].astype(np.float32)
                result[i, j] = np.sum(np.sum(roi * DPM_1 + (1.0 - roi) * DPM_0))
                if result[i, j] > max_prob:
                    best_x, best_y, max_prob = i, j, result[i, j]

        result = result / np.max(np.max(result))
        ruint8 = np.uint8(result * 255)
        ruint8 = cv2.cvtColor(ruint8, cv2.COLOR_GRAY2BGR)
        boxes = []
        for i in range(0, h - model_h, STRIDE):
            for j in range(0, w - model_w, STRIDE):
                if ruint8[i, j, 0] > PROB_THRESHOLD:
                    boxes.append([j, i, j+64, i+192])
                    cv2.rectangle(img=frame_all_bbox,
                                  pt1=(j, i),
                                  pt2=(j + 64, i + 192),
                                  color=(0, 0, 255),
                                  thickness=2)
                    ruint8[i, j, :] = [0, 0, 255]

        boxes = np.array(boxes)
        updated_boxes = non_max_suppression_fast(boxes, 0.4)
        for box in updated_boxes:
            cv2.rectangle(img=frame,
                          pt1=(box[0], box[1]),
                          pt2=(box[2], box[3]),
                          color=(0, 0, 255),
                          thickness=2)

        combined = np.hstack([ruint8, frame_all_bbox, frame])
        # cv2.imshow('Result', ruint8)
        # cv2.imshow('Frame', frame)
        cv2.imshow('left: probabilities; middle: all bboxes; right: suppressed', combined)

        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            break
    else:
        break
#
cv2.destroyAllWindows()
cap.release()
