from plantcv import plantcv as pcv


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


def transform_roi_object():
    print("DEBUG:", "transform roi object not implemented.")
    return None


def transform_analyze_object():
    print("DEBUG:", "transform atalyze object not implemented.")
    return None


def transform_pseudolandmarks():
    print("DEBUG:", "transform pseudolandmarks not implemented.")
    return None
