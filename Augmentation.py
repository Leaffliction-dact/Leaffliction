import skimage as ski
import os
import argparse
import matplotlib.pyplot as plt
from utils.effects import (
    EffectName,
    effect_zoom,
    effect_crop,
    effect_skew,
    effect_rotation,
    effect_blur,
    effect_contrast,
    effect_brightness,
)


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
    parser.add_argument(
        "-g", "--no-gui", action="store_true",
        help="Disable the GUI preview window"
    )
    parser.add_argument(
        "-o", "--output-dir", default="augmented_directory",
        help="Directory to write augmented images into "
             "(default: augmented_directory)"
    )
    parser.add_argument(
        "-w", "--overwrite", action="store_true",
        help="Reprocess images even if their outputs already exist, "
             "overwriting them"
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


CHUNK_SIZE = 50


def get_output_path(filename, effect, base_dir):
    leaf_type = os.path.basename(os.path.dirname(filename))
    out_dir = os.path.join(base_dir, leaf_type)
    name, ext = os.path.splitext(os.path.basename(filename))
    return os.path.join(out_dir, f"{name}_{effect}{ext}")


def is_already_processed(filename, effects, base_dir):
    for effect in effects:
        if (effect == "NONE"):
            continue
        out_path = get_output_path(filename, effect, base_dir)
        if (not os.path.exists(out_path)):
            return False
    return True


def save_augmented_images(
        image_list,
        filenames,
        effects,
        n_rows,
        base_dir,
        overwrite):
    saved = 0
    skipped = 0
    for col, effect in enumerate(effects):
        if (effect == "NONE"):
            continue
        for row in range(n_rows):
            filename = filenames[row]
            out_path = get_output_path(filename, effect, base_dir)
            if (not overwrite and os.path.exists(out_path)):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            image = image_list[row + col*n_rows]
            ski.io.imsave(out_path, image)
            saved += 1
    return saved, skipped


def apply_effects(image_list, effects, n_rows):
    for effect in effects:
        match effect:
            case EffectName.ZOOM.name:
                image_list = effect_zoom(image_list, n_rows, 1.6)
            case EffectName.CROP.name:
                image_list = effect_crop(image_list, n_rows, 1.6)
            case EffectName.SKEW.name:
                image_list = effect_skew(image_list, n_rows, 0.6)
            case EffectName.ROTATION.name:
                image_list = effect_rotation(image_list, n_rows, 0.6)
            case EffectName.BLUR.name:
                image_list = effect_blur(image_list, n_rows, 7.5)
            case EffectName.CONTRAST.name:
                image_list = effect_contrast(image_list, n_rows, 1.5)
            case EffectName.BRIGHTNESS.name:
                image_list = effect_brightness(image_list, n_rows, 0.2)
            case _:
                pass
    return image_list


def process_chunk(filenames, effects, output_dir, overwrite):
    n_effects = sum(1 for effect in effects if effect != "NONE")

    to_process = []
    skipped_images = 0
    for filename in filenames:
        if (not overwrite
                and is_already_processed(filename, effects, output_dir)):
            skipped_images += n_effects
            continue
        to_process.append(filename)

    if (not to_process):
        return [], to_process, skipped_images, 0

    image_list = []
    for filename in to_process:
        image_list.append(ski.io.imread(filename))
    n_rows = len(image_list)
    image_list = apply_effects(image_list, effects, n_rows)
    saved_images, extra_skipped = save_augmented_images(
        image_list, to_process, effects, n_rows, output_dir, overwrite
    )
    skipped_images += extra_skipped
    return image_list, to_process, skipped_images, saved_images


def show_preview(image_list, effects, n_rows):
    n_cols = len(effects)
    max_display_rows = 10
    display_rows = min(n_rows, max_display_rows)
    inches = 2
    fig, axes = plt.subplots(
        display_rows,
        n_cols,
        figsize=(n_cols * inches, display_rows * inches)
    )

    for col in range(n_cols):
        for row in range(display_rows):
            ax = axes[row, col]
            ax.imshow(image_list[row + col*n_rows])
            ax.axis('off')
            if row == 0:
                ax.set_title(effects[col])

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    if (args.effects_help):
        return

    args.effects.insert(0, "NONE")
    total = len(args.input_list)
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    preview = None

    for chunk_idx in range(n_chunks):
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, total)
        chunk_filenames = args.input_list[start:end]

        print(
            f"[{chunk_idx + 1}/{n_chunks}] processing images "
            f"{start + 1}-{end} of {total}..."
        )
        image_list, processed_filenames, skipped, saved = process_chunk(
            chunk_filenames, args.effects, args.output_dir, args.overwrite
        )
        if (skipped):
            print(f"[{chunk_idx + 1}/{n_chunks}] skipped {skipped} "
                  "already-processed images")
        if (saved):
            print(f"[{chunk_idx + 1}/{n_chunks}] saved "
                  f"{saved} images to disk")

        if (preview is None and processed_filenames
                and not args.no_gui):
            preview = (image_list, len(processed_filenames))

    print(f"Done, {total} images processed.")

    if (preview is None):
        return
    preview_images, preview_rows = preview
    show_preview(preview_images, args.effects, preview_rows)


if (__name__ == "__main__"):
    main()
