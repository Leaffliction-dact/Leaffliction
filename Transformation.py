import cv2
import argparse
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils.image_transformations import (
    transform_gaussian_blur,
    transform_mask,
    transform_roi_object,
    transform_analyze_object,
    transform_pseudolandmarks,
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
    mask = transform_mask(original)
    roi_object = transform_roi_object(original)
    analyze_object = transform_analyze_object(original)
    pseudolandmarks = transform_pseudolandmarks(original)
    return (original,
            gaussian_blur,
            mask,
            roi_object,
            analyze_object,
            pseudolandmarks)


def subcommand_show(args):
    transformed_images = transform_image(args.img)
    # build the figure.
    fig = plt.figure(layout="constrained")
    gs = GridSpec(3, 5, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])
    ax7 = fig.add_subplot(gs[:, 2:])
    ax1.set_title("Original")
    ax2.set_title("Gausian blur")
    ax3.set_title("Mask")
    ax4.set_title("Roi objects")
    ax5.set_title("Analyze object")
    ax6.set_title("Pseudolandmarks")
    ax7.set_title("Color histogram")
    # plot all images.
    ax1.imshow(cv2.cvtColor(transformed_images[0], cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(transformed_images[1], cv2.COLOR_BGR2RGB))
    ax3.imshow(cv2.cvtColor(transformed_images[2], cv2.COLOR_BGR2RGB))
    ax4.imshow(cv2.cvtColor(transformed_images[3], cv2.COLOR_BGR2RGB))
    ax5.imshow(cv2.cvtColor(transformed_images[4], cv2.COLOR_BGR2RGB))
    ax6.imshow(cv2.cvtColor(transformed_images[5], cv2.COLOR_BGR2RGB))
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
