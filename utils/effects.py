import cv2
import numpy as np
from enum import Enum
import math


class EffectName(Enum):
    ZOOM = 0
    CROP = 1
    SKEW = 2
    ROTATION = 3
    BLUR = 4
    CONTRAST = 5
    BRIGHTENSS = 6


def effect_zoom(img_arr, rows, zoom_factor):
    for i in range(rows):
        image = img_arr[i].copy()

        h, w = image.shape[:2]
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        cropped = image[top:top+zh, left:left+zw]
        out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        img_arr.append(out)
    print(len(img_arr))
    return img_arr


def effect_crop(img_arr, rows, crop_factor):
    for i in range(rows):
        image = img_arr[i].copy()

        h, w = image.shape[:2]
        ch = int(np.round(h / crop_factor))
        cw = int(np.round(w / crop_factor))
        top = (h - ch) // 2
        left = (w - cw) // 2

        color = np.array([0, 0, 0])
        image[:top, :] = color
        image[h-top:, :] = color
        image[:, :left] = color
        image[:, w-left:] = color

        img_arr.append(image)
    print(len(img_arr))
    return img_arr


def effect_skew(img_arr, rows, skew_factor):
    for i in range(rows):
        image = img_arr[i].copy()

        h, w = image.shape[:2]
        matrix = np.array([
            [1, skew_factor, -skew_factor * h / 2],
            [0, 1,           0],
        ], dtype=np.float32)

        skewed = cv2.warpAffine(image, matrix, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        img_arr.append(skewed)
    print(len(img_arr))
    return img_arr


def effect_rotation(img_arr, rows, rads):
    for i in range(rows):
        image = img_arr[i].copy()

        h, w = image.shape[:2]
        degrees = math.degrees(rads)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)

        rotated = cv2.warpAffine(image, matrix, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
        img_arr.append(rotated)
    print(len(img_arr))
    return img_arr


def effect_blur(img_arr, rows, blurfactor):
    blurfactor /= 100.0
    for i in range(rows):
        image = img_arr[i].copy()

        h, w = image.shape[:2]
        kh = int(blurfactor*h)
        kh += 1 - kh % 2
        kw = int(blurfactor*w)
        kw += 1 - kw % 2
        blurred = cv2.GaussianBlur(image, (kh, kw), 0)
        img_arr.append(blurred)
    print(len(img_arr))
    return img_arr


def effect_contrast(img_arr, rows, factor):
    for i in range(rows):
        image = img_arr[i].copy()

        out = cv2.multiply(image, factor)
        img_arr.append(out)
    print(len(img_arr))
    return img_arr
