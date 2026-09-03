import argparse
from pathlib import Path

from utils.dataset import IMAGE_EXTENSIONS

# per-source: source subdir name -> this repo's canonical class name
SOURCE_CLASS_MAPS = {
    "appleleaf9": {
        "Scab": "Apple_scab",
        "Rust": "Apple_rust",
        "Health": "Apple_healthy",
        # AppleLeaf9 calls this "Frogeye leaf spot"; taxonomically the
        # same bacterium as what this repo's PlantVillage-derived classes
        # call "Apple_Black_rot"
        "Frogeye leaf spot": "Apple_Black_rot",
    },
    "gvlid": {
        "Black rot": "Grape_Black_rot",
        "esca": "Grape_Esca",
        "healthy": "Grape_healthy",
        # GVLiD calls this "leaf blight"; this repo's PlantVillage-derived
        # classes call the equivalent disease "Grape_spot"
        "leaf blight": "Grape_spot",
    },
}

DEFAULT_OUTPUT_DIR = Path("~/goinfre/field_data").expanduser()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Symlink selected class folders from a field-data "
                     "source into this repo's unified class taxonomy, "
                     "producing a directory train.py can take as its "
                     "raw-inputs directory (or --prepared-data source)."
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
             f"(default: {DEFAULT_OUTPUT_DIR}). Reused across sources: "
             "re-running with a different --source adds to it without "
             "touching classes already populated by other sources"
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

        # source-prefixed name: sources numbering their own files from 1
        # (AppleLeaf9's "Scab (1).jpg", a future source's "IMG (1).jpg", ...)
        # would otherwise collide once merged into one canonical class dir
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
            f"[ OK ] {subdir_name!r} -> {canonical_class}: "
            f"{linked} linked, {skipped} already present"
        )

    print(f"Done. Field data available at {args.output_dir}")


if (__name__ == "__main__"):
    main()
