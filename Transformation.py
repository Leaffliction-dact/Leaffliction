import cv2 as cv
import sys
import argparse
from pathlib import Path
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils.image_transformations import (
    transform_gaussian_blur,
    transform_chanel,
    transform_mask,
    transform_roi_boundaries,
    transform_roi_object,
    transform_analyze_object,
    transform_pseudolandmarks,
    transform_normalize,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True

    # argparse subcommand show
    parser_show = subparsers.add_parser("show", help="""display in a window
                                                        the transformations
                                                        preformed on the
                                                        images.""")
    parser_show.add_argument("img", type=str, help="image to be transformed.")

    # argparse subcommand transform
    parser_transform = subparsers.add_parser("transform",
                                             help="""transform a image folder
                                                    with the diferent
                                                    extraction methods.""")
    parser_transform.add_argument("--src",
                                  metavar=('src_directory'),
                                  type=str,
                                  required=True,
                                  help="image folder to be transformed.")
    parser_transform.add_argument("--dst",
                                  metavar=('dst_directory'),
                                  type=str,
                                  required=True,
                                  help="""destination directory for the
                                        transformed images.""")

    args = parser.parse_args()
    return args


def _transform_image(original):
    gaussian_blur = transform_gaussian_blur(original)
    chanel = transform_chanel(gaussian_blur)
    mask = transform_mask(chanel)
    img_stats = transform_roi_boundaries(mask)
    roi_object = transform_roi_object(original, img_stats)
    analyze_object = transform_analyze_object(original, mask)
    pseudolandmarks = transform_pseudolandmarks(original, mask)
    normal_img = transform_normalize(original, mask, img_stats)
    return (gaussian_blur,
            chanel,
            mask,
            roi_object,
            analyze_object,
            pseudolandmarks,
            normal_img)


def _subcommand_show(args):
    original = cv.imread(args.img, cv.IMREAD_COLOR)
    if original == None:
        print("unable to open file as image. check that the filename is correct.")
        sys.exit(1)
    transformed_images = _transform_image(original)
    # build the figure.
    fig = plt.figure(layout="constrained", figsize=(20, 10))
    gs = GridSpec(4, 8, figure=fig, )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, 0])
    ax8 = fig.add_subplot(gs[3, 1])
    ax9 = fig.add_subplot(gs[:, 2:-1])
    ax10 = fig.add_subplot(gs[:, -1])
    ax1.set_title("Original")
    ax2.set_title("Gausian blur")
    ax3.set_title("Chanel")
    ax4.set_title("Mask")
    ax5.set_title("Roi objects")
    ax6.set_title("Analyze object")
    ax7.set_title("Pseudolandmarks")
    ax8.set_title("normalize")
    ax9.set_title("Color histogram")
    # plot all images.
    ax1.imshow(cv.cvtColor(original, cv.COLOR_BGR2RGB))
    ax2.imshow(cv.cvtColor(transformed_images[0], cv.COLOR_BGR2RGB))
    ax3.imshow(cv.cvtColor(transformed_images[1], cv.COLOR_BGR2RGB))
    ax4.imshow(cv.cvtColor(transformed_images[2], cv.COLOR_BGR2RGB))
    ax5.imshow(cv.cvtColor(transformed_images[3], cv.COLOR_BGR2RGB))
    ax6.imshow(cv.cvtColor(transformed_images[4], cv.COLOR_BGR2RGB))
    ax7.imshow(cv.cvtColor(transformed_images[5], cv.COLOR_BGR2RGB))
    ax8.imshow(cv.cvtColor(transformed_images[6], cv.COLOR_BGR2RGB))
    # plot the histogram
    bgr_img = transformed_images[0]
    hsv_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    lab_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2LAB)
    ax9.hist(bgr_img[:, :, 0].flatten(), 256, (0, 255), histtype='step',
             label="blue")
    ax9.hist(bgr_img[:, :, 1].flatten(), 256, (0, 255), histtype='step',
             label="green")
    ax9.hist(bgr_img[:, :, 2].flatten(), 256, (0, 255), histtype='step',
             label="red")
    ax9.hist(hsv_img[:, :, 0].flatten(), 256, (0, 255), histtype='step',
             label="hue")
    ax9.hist(hsv_img[:, :, 1].flatten(), 256, (0, 255), histtype='step',
             label="saturation")
    ax9.hist(hsv_img[:, :, 2].flatten(), 256, (0, 255), histtype='step',
             label="value")
    ax9.hist(lab_img[:, :, 0].flatten(), 256, (0, 255), histtype='step',
             label="lightness")
    ax9.hist(lab_img[:, :, 1].flatten(), 256, (0, 255), histtype='step',
             label="Chrominance-Red")
    ax9.hist(lab_img[:, :, 2].flatten(), 256, (0, 255), histtype='step',
             label="Chrominance-Blue")
    handles, labels = ax9.get_legend_handles_labels()
    ax10.legend(handles, labels, borderaxespad=0, loc=10)
    ax10.axis("off")

    plt.tight_layout()
    plt.show()


def _get_all_filepaths(path):
    if path.exists() == False:
        print("--src path doesn't exist.")
        os.exit(1)
    if path.is_file():
        return [str(path)]
    paths = []
    subdirs = [path]
    while len(subdirs) != 0:
        paths = paths + [x for x in subdirs[0].iterdir() if x.is_file() and x.name[0] != '.']
        if subdirs[0].is_dir():
            subdirs = subdirs[1:] + [x for x in subdirs[0].iterdir() if x.is_dir() and x.name[0] != '.']
        else:
            subdirs = subdirs[1:]
    return paths

    


def _subcommand_transform(args):
    print("DEBUG:\t", "transform subcommand isn't implemented.")
    print("DEBUG:\t", args)
    paths = _get_all_filepaths(Path(args.src))
    for path in paths:
        original = cv.imread(str(path), cv.IMREAD_COLOR)
        if original == None:
            continue
        transformed_images = _transform_image(original)
        
        print(str(path))

    # pcv.print_image(transformed_images, filename="./taha_test2.jpg")


if __name__ == '__main__':
    args = _parse_args()
    if hasattr(args, 'img'):
        _subcommand_show(args)
    else:
        _subcommand_transform(args)
