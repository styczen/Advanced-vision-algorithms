import cv2
import numpy as np
import os

# number of samples per pixel
N = 20

# radius of the sphere
R = 20

# number of close samples for being part of the background
NR_MIN = 2

# amount of random subsampling
PHI = 16

DIR = os.getcwd() + '/../sequences'
SEQUENCE = '/highway'
DIR = DIR + SEQUENCE

TP = 0
TN = 0
FP = 0
FN = 0


def initial_background(gray):
    i_pad = np.pad(gray, 1, 'symmetric')
    height = i_pad.shape[0]
    width = i_pad.shape[1]
    samples = np.zeros((height, width, N))
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            for n in range(N):
                temp_row, temp_col = row, col
                while temp_row == row and temp_col == col:
                    temp_row = np.random.randint(temp_row-1, temp_row+2)
                    temp_col = np.random.randint(temp_col-1, temp_col+2)
                samples[row, col, n] = i_pad[temp_row, temp_col]
    samples = samples[1:height - 1, 1:width - 1]

    return samples


def read_roi_values():
    f = open(DIR + '/temporalROI.txt', 'r')
    line = f.readline()
    roi_start_file, roi_end_file = line.split()
    roi_start_file = int(roi_start_file)
    roi_end_file = int(roi_end_file)
    return roi_start_file, roi_end_file


def get_random_neighborhood(row_idx, col_idx):
    x_n = col_idx
    y_n = row_idx
    while x_n == col_idx and y_n == row_idx:
        x_n = np.random.randint(col_idx - 1, col_idx + 2)
        y_n = np.random.randint(row_idx - 1, row_idx + 2)
    return x_n, y_n


def vibe_detection(gray, samples):
    height = gray.shape[0]
    width = gray.shape[1]
    seg_map = np.zeros((height, width), dtype=np.uint8)
    for row in range(1, height-1):
        for col in range(1, width-1):
            # 1. Compare pixel to background model
            count, index, dist = 0, 0, 0
            while count < NR_MIN and index < N:
                dist = np.abs(gray[row, col] - samples[row, col, index])
                if dist < R:
                    count += 1
                index += 1
            # 2. Classify pixel and update model
            if count >= NR_MIN:
                # 3. Update current pixel model
                # seg_map[row, col] = 0
                r = np.random.randint(0, PHI)
                if r == 0:
                    r = np.random.randint(0, N)
                    samples[row, col, r] = gray[row, col]
                # 4. Update neighboring pixel model
                r = np.random.randint(0, PHI)
                if r == 0:
                    temp_row, temp_col = row, col
                    while temp_row == row and temp_col == col:
                        temp_row = np.random.randint(temp_row-1, temp_row+2)
                        temp_col = np.random.randint(temp_col-1, temp_col+2)
                    samples[temp_row, temp_col, r] = gray[row, col]
            else:
                seg_map[row, col] = 255

    return seg_map, samples


if __name__ == '__main__':
    roi_start, roi_end = read_roi_values()

    kernel = np.ones((5, 5), np.uint8)

    # current image
    image = cv2.imread(DIR + '/input/in%06d.jpg' % roi_start, cv2.IMREAD_GRAYSCALE)

    # background model
    bg_model = initial_background(image)

    for i in range(roi_start, roi_end):
        current_img_color = cv2.imread(DIR+'/input/in%06d.jpg' % i)
        current_img = current_img_color.copy()
        current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        mask, bg_model = vibe_detection(current_img, bg_model)

        cv2.imshow('Mask', mask)

        mask = cv2.medianBlur(mask, 5)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        cv2.imshow('Morphed mask', mask)

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        cv2.imshow("Labels", np.uint8(labels / stats.shape[0] * 255))

        if stats.shape[0] > 1:      # czy sa jakies obiekty
            tab = stats[1:, 4]      # wyciecie 4 kolumny bez pierwszego elementu
            pi = np.argmax(tab)     # znalezienie indeksu najwiekszego elementu
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
                        color=(0, 255, 0))

        cv2.imshow('Labeled frame', current_img_color)

        # Evaluate
        gt_img = cv2.imread(DIR+'/groundtruth/gt%06d.png' % i)

        # cv2.imshow('gt', gt_img)

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

        print('Frame {}'.format(i))

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = 2*P*R/(P + R)

    print('P:', P)
    print('R:', R)
    print('F1:', F1)

    cv2.destroyAllWindows()
