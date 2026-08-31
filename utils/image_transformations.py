from plantcv import plantcv as pcv
import cv2 as cv
import numpy as np


def transform_gaussian_blur(img):
    img = img.copy()
    img = pcv.gaussian_blur(img, (13, 13), 0)

    return img


def transform_chanel(img):
    img = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    chanel = img[:, :, 2]

    return chanel


def transform_mask(img):
    img = img.copy()
    _, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    interior = img.copy()
    h, w = interior.shape[:2]
    tmp_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(interior, tmp_mask, (0, 0), 255)
    interior = cv.bitwise_not(interior)

    img = img | interior

    return img


def transform_roi_boundaries(mask):
    _, _, stats, _ = cv.connectedComponentsWithStats(mask, 8, cv.CV_32S)
    best = stats[1]
    for stat in stats[1:]:
        if stat[-1] > best[-1]:
            best = stat
    best = best[:-1]
    return best


def transform_roi_object(img, stats):
    img = img.copy()
    img = cv.rectangle(img,
                       (stats[0], stats[1]),
                       (stats[0] + stats[2], stats[1] + stats[3]),
                       (255, 0, 0),
                       3)

    return img


def transform_analyze_object(img, mask):
    img = img.copy()
    img = pcv.analyze.size(img=img, labeled_mask=mask, n_labels=1)
    return img


def _render_landmarks(img, landmarks, color):
    for landmark in landmarks:
        x, y = landmark[0]
        cv.circle(img,
                  (int(x), int(y)),
                  radius=5,
                  color=color,
                  thickness=-1)


def transform_pseudolandmarks(img, mask):
    img = img.copy()
    top, bottom, center_v = pcv.homology.y_axis_pseudolandmarks(img=img,
                                                                mask=mask)
    _render_landmarks(img, top, (255, 0, 0))
    _render_landmarks(img, bottom, (0, 255, 0))
    _render_landmarks(img, center_v, (0, 0, 255))
    return img


def transform_normalize(img, mask, stats):
    img = img.copy()
    img = cv.bitwise_or(img, img, mask=mask)
    img[mask == 0] = (255, 255, 255)

    img = img[stats[1]:stats[1] + stats[3], stats[0]:stats[0] + stats[2]]
    img = cv.resize(img, (128, 128))

    return img
