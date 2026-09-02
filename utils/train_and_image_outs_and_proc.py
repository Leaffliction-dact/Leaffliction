import json
import random
import zipfile
import numpy as np
import cv2
from pathlib import Path
from plantcv import plantcv as pcv
from Augmentation import apply_effects
from utils.effects import EffectName
from utils.image_transformations import (
    transform_gaussian_blur,
    transform_chanel,
    transform_mask,
    transform_roi_boundaries,
    transform_normalize,
)

pcv.params.verbose = False

# start small for iteration speed try 224/256 later
# btw, 64 apparently works great
INPUT_SIZE = 128
AUGS_TO_MAKE = 3


def mask_and_resize(img: np.ndarray, size=INPUT_SIZE) -> np.ndarray:
    gaussian_blur = transform_gaussian_blur(img)
    channel = transform_chanel(gaussian_blur)
    mask = transform_mask(channel)
    img_stats = transform_roi_boundaries(mask)
    return transform_normalize(img, mask, img_stats, size=size)


def _apply_effect_sequence(img: np.ndarray, effect_names: list) -> np.ndarray:
    for name in effect_names:
        img = apply_effects([img], [name], 1)[1]
    return img


# this wrties down prepared images, effects applied or not
def _prepare_and_write_imgs(
        raw_path: Path,
        out_dir: Path,
        inpsize: int,
        effect_names=None,
        i=None,
        ):
    img = cv2.imread(str(raw_path))

    if (effect_names):
        img = _apply_effect_sequence(img, effect_names)
    img = mask_and_resize(img, inpsize)

    if (effect_names):
        tag = "_".join(effect_names)
    else:
        tag = "orig"
    if (i is not None):
        out_path = out_dir / f"{raw_path.stem}_{tag}_{i}.jpg"
    else:
        out_path = out_dir / f"{raw_path.stem}_{tag}.jpg"
    cv2.imwrite(str(out_path), img)
    return out_path


# this orders the prepare imgs what kinds of augs to write down
def _augment_base_image(
        raw_path: Path,
        class_dir: Path,
        effect_names: list,
        inpsize: int,
        label: int):
    augmented = []

    print("writing... ", end="")
    for i in range(AUGS_TO_MAKE):
        seq_effects = ["ROTATION"]
        seq_effects.extend(random.sample(effect_names, 2))
        p = _prepare_and_write_imgs(
            raw_path,
            class_dir,
            effect_names=seq_effects,
            inpsize=inpsize,
            i=i
        )
        augmented.append((p, label))
        print(f"{i:2d}", end="")
    print()

    return augmented


# prepares one of the splits, those being training or validation sets
def prepare_split(
        samples,
        class_to_idx,
        out_dir: Path,
        needs_augmenting: bool,
        inpsize: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = []

    class_paths = {}
    for raw_path, cls in samples:
        class_paths.setdefault(cls, []).append(raw_path)

    effect_names = [effect.name for effect in EffectName]

    total = len(samples)
    processed = 0
    for cls, raw_paths in class_paths.items():
        class_dir = out_dir / cls
        class_dir.mkdir(exist_ok=True)
        label = class_to_idx[cls]

        for raw_path in raw_paths:
            p = _prepare_and_write_imgs(raw_path, class_dir, inpsize=inpsize)
            prepared.append((p, label))
            processed += 1
            print(f"  [{processed:5d}/{total}] {cls}/{raw_path.name}")

        if (needs_augmenting):
            processed_augs = 0
            total_augs = AUGS_TO_MAKE * len(raw_paths)
            print("             And the augs...")
            for raw_path in raw_paths:
                augmented = _augment_base_image(
                    raw_path,
                    class_dir,
                    effect_names,
                    inpsize,
                    label
                )
                prepared.extend(augmented)
                processed_augs += AUGS_TO_MAKE
                print(
                    f"      [{processed_augs:5d}/{total_augs}]"
                    f" {cls}/{raw_path.name}"
                )
            print(f"    +{len(raw_paths) * AUGS_TO_MAKE} augmented for {cls}")

    return prepared


def write_outs(
        class_to_idx,
        train_dir: Path,
        input_size: int,
        model_path: Path,
        zip_path: Path,
        class_map_path: Path,
        img_dim_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    class_map_path.parent.mkdir(parents=True, exist_ok=True)
    img_dim_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = train_dir.parent.parent
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, arcname=model_path.name)

        with open(class_map_path, "w") as f:
            json.dump(class_to_idx, f)
        zf.write(class_map_path, arcname=class_map_path.name)

        with open(img_dim_path, "w") as f:
            f.write(str(input_size))
        zf.write(img_dim_path, arcname=img_dim_path.name)

        for path in train_dir.rglob("*.jpg"):
            zf.write(path, arcname=str(path.relative_to(base_dir)))


def write_plain_outs(
        class_to_idx,
        input_size: int,
        class_map_path: Path,
        img_dim_path: Path):
    class_map_path.parent.mkdir(parents=True, exist_ok=True)
    img_dim_path.parent.mkdir(parents=True, exist_ok=True)

    with open(class_map_path, "w") as f:
        json.dump(class_to_idx, f)

    with open(img_dim_path, "w") as f:
        f.write(str(input_size))
