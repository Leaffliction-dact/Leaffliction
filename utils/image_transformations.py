from plantcv import plantcv as pcv
import cv2
import numpy as np


def transform_gaussian_blur(img):
    img = pcv.gaussian_blur(img, (41, 41), sigma_x=0, sigma_y=0)
    return img
    # print("DEBUG:", "transform gausian blur not implemented.")


def transform_mask(img):
    img = transform_gaussian_blur(img)
    img = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    #   calculating the threshold by piking a point in the midle of the
    #   values of the center of the plant and an edge of background.
    heuristic_threashold = min(img[128, 128], img[255, 0]) + \
        abs(img[128, 128] - img[255, 0]) / 3
    img = pcv.threshold.binary(gray_img=img,
                               threshold=heuristic_threashold,
                               object_type="light")
    return img


def transform_roi_object(img):
    mask = transform_mask(img)
    img = img.copy()
    contours, hierarchy = cv2.findContours(mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 10)
    mask = cv2.merge([mask, mask, mask])
    green = np.full_like(img, (0, 255, 0))
    img = np.where(mask == 0, img, green)
    return img


def transform_analyze_object(img):
    mask = transform_mask(img)
    img = pcv.analyze.size(img=img, labeled_mask=mask, n_labels=1)
    return img


def transform_pseudolandmarks():
    print("DEBUG:", "transform pseudolandmarks not implemented.")
    return None
