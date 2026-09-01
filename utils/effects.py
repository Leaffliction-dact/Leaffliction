import cv2
import numpy as np
from enum import Enum
import math
import random


class EffectName(Enum):
    ZOOM = 0
    CROP = 1
    SKEW = 2
    ROTATION = 3
    BLUR = 4
    CONTRAST = 5
    BRIGHTNESS = 6


def effect_zoom(img_arr, rows, zoom_factor):
    if (zoom_factor < 1.1):
        zoom_factor = 1.2
    for i in range(rows):
        zf = random.uniform(1.1, zoom_factor)
        image = img_arr[i].copy()

        h, w = image.shape[:2]
        zh = int(np.round(h / zf))
        zw = int(np.round(w / zf))
        top = (h - zh) // 2
        left = (w - zw) // 2

        cropped = image[top:top+zh, left:left+zw]
        out = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        img_arr.append(out)
    return img_arr


def effect_crop(img_arr, rows, crop_factor):
    if (crop_factor < 1.1):
        crop_factor = 1.2
    for i in range(rows):
        cf = random.uniform(1.1, crop_factor)
        image = img_arr[i].copy()

        h, w = image.shape[:2]
        ch = int(np.round(h / cf))
        cw = int(np.round(w / cf))
        top = (h - ch) // 2
        left = (w - cw) // 2

        color = np.array([255, 255, 255])
        image[:top, :] = color
        image[h-top:, :] = color
        image[:, :left] = color
        image[:, w-left:] = color

        img_arr.append(image)
    return img_arr


def effect_skew(img_arr, rows, skew_factor):
    if (skew_factor < 0):
        skew_factor = 0.3
    for i in range(rows):
        sf = random.uniform(0, skew_factor)
        if (random.random() >= 0.5):
            sf *= -1
        image = img_arr[i].copy()

        h, w = image.shape[:2]
        matrix = np.array([
            [1, sf, -sf * h / 2],
            [0, 1,            0],
        ], dtype=np.float32)

        skewed = cv2.warpAffine(image, matrix, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        img_arr.append(skewed)
    return img_arr


def effect_rotation(img_arr, rows, rads):
    if (rads < 1):
        rads = 1
    for i in range(rows):
        rs = random.uniform(0, rads)
        if (random.random() >= 0.5):
            rs *= -1
        image = img_arr[i].copy()

        h, w = image.shape[:2]
        degrees = math.degrees(rs)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)

        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        img_arr.append(rotated)
    return img_arr


def effect_blur(img_arr, rows, blurfactor):
    if (blurfactor < 5):
        blurfactor = 5
    for i in range(rows):
        image = img_arr[i].copy()
        bf = random.uniform(3, blurfactor)
        bf = bf / 100

        h, w = image.shape[:2]
        kh = int(bf*h)
        kh += 1 - kh % 2
        kw = int(bf*w)
        kw += 1 - kw % 2
        blurred = cv2.GaussianBlur(image, (kh, kw), 0)
        img_arr.append(blurred)
    return img_arr


def effect_contrast(img_arr, rows, factor):
    contrast_range = abs(factor - 1.0)
    if (contrast_range < 0.2):
        contrast_range = 0.2
    elif (contrast_range > 0.8):
        contrast_range = 0.8
    for i in range(rows):
        cr = random.uniform(0.1, contrast_range)
        if (random.random() >= 0.5):
            cr *= -1
        cr += 1.2
        image = img_arr[i].copy()

        beta = 128 * (1 - cr)
        out = cv2.addWeighted(image, cr, image, 0, beta)
        img_arr.append(out)
    return img_arr


def effect_brightness(img_arr, rows, factor):
    f = abs(factor)
    if (f < 0.2):
        f = 0.2
    for i in range(rows):
        ff = random.uniform(0.1, f)
        if (random.random() >= 0.5):
            ff *= -1
        ff = 255*ff
        ff = int(ff)
        image = img_arr[i].copy()

        out = cv2.add(image, ff)
        img_arr.append(out)
    return img_arr
