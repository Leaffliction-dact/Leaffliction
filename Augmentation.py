import skimage as ski
import os
import argparse
import matplotlib.pyplot as plt
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
    if (not args.effects):
        args.effects = [effect.name for effect in EffectName]
    return args


def main():
    args = parse_args()
    image_list = []
    for filename in args.input_list:
        image_list.append(ski.io.imread(filename))
    print(image_list)
    print(args)

    n_rows = len(image_list)
    n_cols = len(args.effects)
    inches = 2
    fig, axes = plt.subplots(n_rows, n_cols,
            figsize=(n_cols * inches, n_rows * inches))

    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.imshow(image_list[row])
            ax.axis('off')
            if row == 0:
                ax.set_title(args.effects[col])

    plt.tight_layout()
    plt.show()


if (__name__ == "__main__"):
    main()
