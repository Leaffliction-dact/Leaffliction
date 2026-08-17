import cv2 as cv
import sys
import matplotlib.pyplot as plt 
import numpy as np


def plot_original(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    plt.figure("original")
    plt.imshow(img)
    

def chanel(img):
    print(type(img))
    print(img.shape)
    img = cv.GaussianBlur(img, (13, 13), 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    chanel = img[:, :, 2]

    plt.figure("chanel")
    plt.imshow(chanel)
    plt.colorbar()

    return chanel


def plot_histogram(chanel):
    
    plt.figure("histogram")
    count, bins = np.histogram(chanel, 256)
    plt.stairs(count, bins)


def threashold(chanel):
    _, threashold = cv.threshold(chanel, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    #mask = cv.bitwise_not(mask)

    plt.figure("binary mask")
    plt.imshow(threashold)

    return mask


def test1(mask):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, 8, cv.CV_32S)
    print("test1:\t", num_labels)
    print("test2:\t", labels)
    print("test4:\t", stats)
    print("test5:\t", centroids)
    
    plt.figure("connected_components")
    plt.imshow(labels)
    plt.colorbar()


def flood_fill_mask(threashold):
    floodfill = threashold.copy()
    
    h, w = threashold.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv.floodFill(floodfill, mask, (0, 0), 255)
    floodfill = cv.bitwise_not(floodfill)
    final_mask = threashold | floodfill
    
    plt.figure("floodfill")
    plt.imshow(final_mask)

    return final_mask


def mask_img(mask, img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.bitwise_or(img, img, mask=mask)
    img[mask==0] = (255, 255, 255)
    
    plt.figure("masked image")
    plt.imshow(img)

    return img


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("please provide only one image.")
        sys.exit()
    img = cv.imread(sys.argv[1])
    plot_original(img)
    chanel = chanel(img)
    plot_histogram(chanel)
    threashold = threashold(chanel)
    mask = flod_fill_mask(mask)
    test1(mask)
    mask_img(mask, img)
    plt.show()
