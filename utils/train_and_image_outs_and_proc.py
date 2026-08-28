import json
import random
import warnings
import zipfile
import numpy as np
import cv2
from pathlib import Path
from plantcv import plantcv as pcv
from Augmentation import apply_effects
from utils.effects import EffectName

pcv.params.verbose = False
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"plantcv\.plantcv\.(fill|closing)")

# start small for iteration speed try 224/256 later
# btw, 64 apparently works great
INPUT_SIZE = 128
MASK_CROP_PADDING = 10
MASKING_MAX_DIM = 512
MASK_REFERENCE_DIM = 256
MASK_FILL_MIN_AREA = 200
MASK_CLOSE_KERNEL_SIZE = 15

# this much of the originals will be preserved (the rest will go through augs)
AUGMENT_BASE_FRACTION = 0.25


def _crop_to_largest_component(
        img: np.ndarray,
        mask: np.ndarray,
        padding=MASK_CROP_PADDING) -> tuple:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if (num_labels <= 1):
        return img, mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    component_mask = np.where(labels == largest_label, mask, 0)
    component_mask = component_mask.astype(np.uint8)

    x = stats[largest_label, cv2.CC_STAT_LEFT]
    y = stats[largest_label, cv2.CC_STAT_TOP]
    w = stats[largest_label, cv2.CC_STAT_WIDTH]
    h = stats[largest_label, cv2.CC_STAT_HEIGHT]

    img_h, img_w = mask.shape[:2]
    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, img_w)
    y1 = min(y + h + padding, img_h)

    return img[y0:y1, x0:x1], component_mask[y0:y1, x0:x1]


def mask_and_resize(img: np.ndarray, size=INPUT_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if (max_dim > MASKING_MAX_DIM):
        downscale = MASKING_MAX_DIM / max_dim
        img = cv2.resize(
            img,
            (int(w * downscale), int(h * downscale)),
            interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    area_scale = (h * w) / (MASK_REFERENCE_DIM ** 2)
    linear_scale = area_scale ** 0.5

    a_channel = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    green_mask = pcv.threshold.otsu(gray_img=a_channel, object_type="dark")

    b_channel = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    yellow_mask = pcv.threshold.otsu(gray_img=b_channel, object_type="light")

    mask = pcv.logical_or(bin_img1=green_mask, bin_img2=yellow_mask)
    fill_min_area = int(MASK_FILL_MIN_AREA * area_scale)
    mask = pcv.fill(bin_img=mask, size=fill_min_area)

    close_size = int(MASK_CLOSE_KERNEL_SIZE * linear_scale)
    if (close_size % 2 == 0):
        close_size += 1
    close_size = max(close_size, 3)
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size)
    )
    mask = pcv.closing(gray_img=mask, kernel=close_kernel)
    mask = pcv.fill_holes(bin_img=mask)

    img, mask = _crop_to_largest_component(img, mask)

    masked = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    return cv2.resize(masked, (size, size))


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
        ):
    img = cv2.imread(str(raw_path))

    if (effect_names):
        img = _apply_effect_sequence(img, effect_names)
    img = mask_and_resize(img, inpsize)

    if (effect_names):
        tag = "_".join(effect_names)
    else:
        tag = "orig"
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

    solo_effects = random.sample(effect_names, 2)
    for name in solo_effects:
        p = _prepare_and_write_imgs(
            raw_path,
            class_dir,
            effect_names=[name],
            inpsize=inpsize
        )
        augmented.append((p, label))

    seq_effects = random.sample(effect_names, 2)
    p = _prepare_and_write_imgs(
        raw_path,
        class_dir,
        effect_names=seq_effects,
        inpsize=inpsize
    )
    augmented.append((p, label))

    return augmented


def _select_base_imgs_for_augs(raw_paths: list, target_count: int) -> list:
    base_size = int(target_count * AUGMENT_BASE_FRACTION)
    if (len(raw_paths) < base_size):
        return raw_paths
    return random.sample(raw_paths, base_size)


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

    if (needs_augmenting):
        max_count = max(len(paths) for paths in class_paths.values())

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
            print(f"  [{processed:4d}/{total}] {cls}/{raw_path.name}")

        if (needs_augmenting):
            base_paths = _select_base_imgs_for_augs(raw_paths, max_count)
            for base_path in base_paths:
                augmented = _augment_base_image(
                    base_path,
                    class_dir,
                    effect_names,
                    inpsize,
                    label
                )
                prepared.extend(augmented)
            print(f"    +{len(base_paths) * 3} augmented for {cls}")

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
