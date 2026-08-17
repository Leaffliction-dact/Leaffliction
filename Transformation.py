import cv2 as cv
import argparse
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


def parse_args():
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


def transform_image(img_filename):
    original, _, _ = pcv.readimage(filename=img_filename)
    gaussian_blur = transform_gaussian_blur(original)
    chanel = transform_chanel(gaussian_blur)
    mask = transform_mask(chanel)
    img_stats = transform_roi_boundaries(mask)
    roi_object = transform_roi_object(original, img_stats)
    analyze_object = transform_analyze_object(original, mask)
    pseudolandmarks = transform_pseudolandmarks(original, mask)
    normal_img = transform_normalize(original, mask, img_stats)
    return (original,
            gaussian_blur,
            chanel,
            mask,
            roi_object,
            analyze_object,
            pseudolandmarks,
            normal_img)


def subcommand_show(args):
    transformed_images = transform_image(args.img)
    # build the figure.
    fig = plt.figure(layout="constrained")
    gs = GridSpec(4, 6, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[3, 0])
    ax8 = fig.add_subplot(gs[3, 1])
    ax9 = fig.add_subplot(gs[:, 2:])
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
    ax1.imshow(cv.cvtColor(transformed_images[0], cv.COLOR_BGR2RGB))
    ax2.imshow(cv.cvtColor(transformed_images[1], cv.COLOR_BGR2RGB))
    ax3.imshow(cv.cvtColor(transformed_images[2], cv.COLOR_BGR2RGB))
    ax4.imshow(cv.cvtColor(transformed_images[3], cv.COLOR_BGR2RGB))
    ax5.imshow(cv.cvtColor(transformed_images[4], cv.COLOR_BGR2RGB))
    ax6.imshow(cv.cvtColor(transformed_images[5], cv.COLOR_BGR2RGB))
    ax7.imshow(cv.cvtColor(transformed_images[6], cv.COLOR_BGR2RGB))
    ax8.imshow(cv.cvtColor(transformed_images[7], cv.COLOR_BGR2RGB))

    plt.show()


def subcommand_transform(args):
    print("DEBUG:\t", "transform subcommand isn't implemented.")
    # pcv.print_image(transformed_images, filename="./taha_test2.jpg")


if __name__ == '__main__':
    args = parse_args()
    print("DEBUG:\t", args)
    if hasattr(args, 'img'):
        subcommand_show(args)
    else:
        subcommand_transform(args)
