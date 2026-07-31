import skimage as ski
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform an input image in various distinct ways."
    )
    parser.add_argument(
        "-i", "--input-list", nargs="+", required=True,
        help="A list of images to process (minimum one)"
    )
    parser.add_argument(
        "-e", "--effects", nargs="+",
        help="Choose some effects to apply (instead of ALL by default)"
    )
    parser.add_argument(
        "-E", "--effects-help", action="store_true",
        help="Print the list of effects available for the -e option"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)


if (__name__ == "__main__"):
    main()
