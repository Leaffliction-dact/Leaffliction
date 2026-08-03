import skimage as ski
import os
import argparse
from utils.effects import EffectName


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform an input image in various distinct ways."
    )
    parser.add_argument(
        "-i", "--input-list", nargs="+",
        help="A list of images to process (minimum one)"
    )
    parser.add_argument(
        "-e", "--effects", nargs="+",
        choices=[effect.name for effect in EffectName],
        help="Choose some effects to apply (instead of ALL by default)"
    )
    parser.add_argument(
        "-E", "--effects-help", action="store_true",
        help="Print the list of effects available for the -e option"
    )
    args = parser.parse_args()
    if (not args.input_list and not args.effects_help):
        parser.error("Either enter a list of input images with -i or "
                     "ask for the list of available effects with -E")
    elif (args.effects_help):
        effect_names = [effect.name for effect in EffectName]
        print("Available effects:")
        for effect_name in effect_names:
            print(f"  {effect_name}")
    return args


def main():
    args = parse_args()
    print(args)


if (__name__ == "__main__"):
    main()
