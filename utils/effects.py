import numpy as np
from enum import Enum
from scipy.ndimage import zoom


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

        zoom_tuple = (zoom_factor,) * 2 + (1,) * (image.ndim - 2)
        h, w = image.shape[:2]
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(image[top:top+zh, left:left+zw], zoom_tuple)

        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]
        img_arr.append(out)
    print(len(img_arr))
    return img_arr


def effect_crop(img_arr, rows, crop_factor):
    for i in range(rows):
        image = img_arr[i].copy()
        print(image.shape)

        h, w = image.shape[:2]
        ch = int(np.round(h / crop_factor))
        cw = int(np.round(w / crop_factor))
        top = (h - ch) // 2
        left = (w - cw) // 2

        color = np.array([0, 0, 0])
        image[:top, :] = color
        image[-top:, :] = color
        image[:, :left] = color
        image[:, -left:] = color

        img_arr.append(image)
    print(len(img_arr))
    return img_arr
