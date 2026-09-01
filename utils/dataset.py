from pathlib import Path

IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp")


def discover_class_images(root: Path) -> dict:
    """
    Maps each subdir of root to its image files. Each
    subdir is one class (e.g. Apple_healthy, Grape_Black_rot).
    Casing varies
    """
    class_images = {}
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(item.glob(f"*.{ext}"))
            images.extend(item.glob(f"*.{ext.upper()}"))
        class_images[item.name] = sorted(images)
    return class_images
