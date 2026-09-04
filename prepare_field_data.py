import argparse
from pathlib import Path

from utils.dataset import IMAGE_EXTENSIONS

SOURCE_CLASS_MAPS = {
    "appleleaf9": {
        "Scab": "Apple_scab",
        "Rust": "Apple_rust",
        "Health": "Apple_healthy",
        "Frogeye leaf spot": "Apple_Black_rot",
    },
    "gvlid": {
        "Black rot": "Grape_Black_rot",
        "esca": "Grape_Esca",
        "healthy": "Grape_healthy",
        "leaf blight": "Grape_spot",
    },
}

DEFAULT_OUTPUT_DIR = Path("~/goinfre/field_data").expanduser()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Symlink selected class folders from the field data"
                    "into this repo's class taxonomy for use in train.py"
    )
    parser.add_argument(
        "source_root", type=Path,
        help="Root directory of the source dataset "
             "(e.g. ~/goinfre/AppleLeaf9)"
    )
    parser.add_argument(
        "-s", "--source", choices=sorted(SOURCE_CLASS_MAPS), required=True,
        help="Which source's class-name mapping to apply"
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Unified field-data directory to symlink into "
             f"(default: {DEFAULT_OUTPUT_DIR}). Re-use with different sources"
    )
    args = parser.parse_args()

    if (not args.source_root.is_dir()):
        parser.error(f"Source root not found: {args.source_root}")

    return args


def symlink_class_images(source_dir: Path, dest_dir: Path, source_key: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    skipped = 0

    for image_path in sorted(source_dir.iterdir()):
        if (not image_path.is_file()):
            continue
        if (image_path.suffix.lower().lstrip(".") not in IMAGE_EXTENSIONS):
            continue

        link_path = dest_dir / f"{source_key}__{image_path.name}"
        if (link_path.exists() or link_path.is_symlink()):
            skipped += 1
            continue

        link_path.symlink_to(image_path.resolve())
        linked += 1

    return linked, skipped


def main():
    args = parse_args()
    class_map = SOURCE_CLASS_MAPS[args.source]

    for subdir_name, canonical_class in class_map.items():
        source_dir = args.source_root / subdir_name
        if (not source_dir.is_dir()):
            print(f"[WARN] missing source class dir: {source_dir}")
            continue

        dest_dir = args.output_dir / canonical_class
        linked, skipped = symlink_class_images(
            source_dir, dest_dir, args.source
        )
        print(
            f"[ OK ] {subdir_name!r} into {canonical_class}: "
            f"{linked} linked, {skipped} already present"
        )

    print(f"Done. Field data available at {args.output_dir}")


if (__name__ == "__main__"):
    main()
