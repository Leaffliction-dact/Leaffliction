import argparse
import json
import random
import warnings
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from plantcv import plantcv as pcv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from Augmentation import apply_effects
from utils.dataset import discover_class_images
from utils.effects import EffectName
from leafcnn import LeafCNN
from leafset import LeafDataset

pcv.params.verbose = False
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"plantcv\.plantcv\.(fill|closing)")

# start small for iteration speed try 224/256 later
# btw, 64 apparently works great
INPUT_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
DROPOUT_P = 0.3

MAX_EPOCHS = 50
PATIENCE = 7

# this much of the originals will be preserved (the rest will go through augs)
AUGMENT_BASE_FRACTION = 0.25

# seemingly better for 'realism' and prediction later.
AUGMENT_BEFORE_MASK = True

MASK_REFERENCE_DIM = 256
MASK_FILL_MIN_AREA = 200
MASK_CLOSE_KERNEL_SIZE = 15
MASKING_MAX_DIM = 512
MASK_CROP_PADDING = 10

DEFAULT_CACHE_DIR = Path("~/goinfre/_prepared_data").expanduser()
DEFAULT_MODEL_PATH = Path("best_model.pt")
DEFAULT_ZIP_PATH = Path("~/goinfre/leafzip/learnings.zip").expanduser()
DEFAULT_CLASS_MAP_PATH = Path("class_to_idx.json")
DEFAULT_IMG_DIM_PATH = Path("img_dim")


def discover_classes(root: Path, max_images_per_class=None):
    class_images = discover_class_images(root)
    class_to_idx = {
        name: i for i, name in enumerate(sorted(class_images))
    }

    samples = []
    for class_name, images in class_images.items():
        if (max_images_per_class is not None):
            images = images[:max_images_per_class]
        for path in images:
            samples.append((path, class_name))

    return samples, class_to_idx


def discover_prepared_classes(prepared_dir: Path):
    class_images = discover_class_images(prepared_dir / "train")
    class_to_idx = {
        name: i for i, name in enumerate(sorted(class_images))
    }
    return class_to_idx


def load_prepared_split(split_dir: Path, class_to_idx: dict):
    class_images = discover_class_images(split_dir)
    samples = []
    for class_name, images in class_images.items():
        label = class_to_idx[class_name]
        for path in images:
            samples.append((path, label))
    return samples


def _crop_to_largest_component(
        img: np.ndarray,
        mask: np.ndarray,
        padding=MASK_CROP_PADDING) -> tuple:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
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


# also use in predict.py
# don't forget to get the masking from t's version
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


def augmentation_base(raw_paths: list, target_count: int) -> list:
    base_size = int(target_count * AUGMENT_BASE_FRACTION)
    if (len(raw_paths) < base_size):
        return raw_paths
    return random.sample(raw_paths, base_size)


def _apply_effect_sequence(img: np.ndarray, effect_names: list) -> np.ndarray:
    for name in effect_names:
        img = apply_effects([img], [name], 1)[1]
    return img


def _process_and_save(
        raw_path: Path,
        out_dir: Path,
        inpsize: int,
        effect_names=None,
        ):
    img = cv2.imread(str(raw_path))

    if (not effect_names):
        img = mask_and_resize(img, inpsize)
    elif (AUGMENT_BEFORE_MASK):
        img = _apply_effect_sequence(img, effect_names)
        img = mask_and_resize(img, inpsize)
    else:
        img = mask_and_resize(img, inpsize)
        img = _apply_effect_sequence(img, effect_names)

    if (effect_names):
        tag = "_".join(effect_names)
    else:
        tag = "orig"
    out_path = out_dir / f"{raw_path.stem}_{tag}.jpg"
    cv2.imwrite(str(out_path), img)
    return out_path


def _augment_base_image(
        raw_path: Path,
        class_dir: Path,
        effect_names: list,
        inpsize: int,
        label: int):
    augmented = []

    solo_effects = random.sample(effect_names, 2)
    for name in solo_effects:
        p = _process_and_save(
            raw_path,
            class_dir,
            effect_names=[name],
            inpsize=inpsize
        )
        augmented.append((p, label))

    seq_effects = random.sample(effect_names, 2)
    p = _process_and_save(
        raw_path,
        class_dir,
        effect_names=seq_effects,
        inpsize=inpsize
    )
    augmented.append((p, label))

    return augmented


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
            p = _process_and_save(raw_path, class_dir, inpsize=inpsize)
            prepared.append((p, label))
            processed += 1
            print(f"  [{processed:4d}/{total}] {cls}/{raw_path.name}")

        if (needs_augmenting):
            base_paths = augmentation_base(raw_paths, max_count)
            for raw_path in base_paths:
                augmented = _augment_base_image(
                    raw_path,
                    class_dir,
                    effect_names,
                    inpsize,
                    label
                )
                prepared.extend(augmented)
            print(f"    +{len(base_paths) * 3} augmented for {cls}")

    return prepared


def build_class_weights(train_samples, class_to_idx, device):
    counts = {idx: 0 for idx in class_to_idx.values()}
    for _, label in train_samples:
        counts[label] += 1

    total = sum(counts.values())
    num_classes = len(class_to_idx)
    ordered_counts = [counts[i] for i in range(num_classes)]

    weights = torch.tensor(
        [total / (num_classes * c) for c in ordered_counts],
        dtype=torch.float32,
    )
    return weights.to(device)


def evaluate(model, loader, criterion, device):
    correct, total = 0, 0
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total, total_loss / total


def train(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        device,
        max_epochs,
        patience,
        model_path: Path):
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for epoch in range(max_epochs):
            model.train()
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_acc, val_loss = evaluate(
                    model, val_loader, criterion, device
                )

            print(
                f"epoch {epoch + 1:4d}/{max_epochs}  "
                f"val_acc={val_acc:.4f}  val_loss={val_loss:.4f}"
            )

            if (val_acc > best_val_acc):
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_path)

            if (val_loss < best_val_loss):
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (epochs_without_improvement >= patience):
                print(f"no val_loss improvement for {patience} epochs, "
                      "stopping early")
                break
    except KeyboardInterrupt:
        print("Stopping & saving best results")

    return best_val_acc


def package_outputs(
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory", type=Path, nargs="?", default=None,
        help="Path to raw inputs. Ignored with --prepared-data"
    )
    parser.add_argument(
        "-p", "--prepared-data", type=Path, default=None,
        help="Path to an already-pre-prepared dataset. "
             "Has to be of train-val format (like what this program "
             "produces if ran without the option)"
    )
    parser.add_argument(
        "-n", "--max-images-per-class", type=int, default=None,
        help="Cap the number of input images read per class "
             "(default: none). Ignored with --prepared-data"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=MAX_EPOCHS,
        help=f"Maximum number of training epochs (default: {MAX_EPOCHS})"
    )
    parser.add_argument(
        "-d", "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device to train on (default: cpu)"
    )
    parser.add_argument(
        "-D", "--dropout", type=float, default=DROPOUT_P,
        help=f"Dropout percentage (0.0 to 0.9, default {DROPOUT_P})"
    )
    parser.add_argument(
        "-I", "--input-size", type=int, default=INPUT_SIZE,
        help="The size to which the inputs will be scaled, in pixels, "
             "one side only (since the inputs are squares). "
             f"Default is {INPUT_SIZE}, ranges 16..512."
    )
    parser.add_argument(
        "-B", "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (2..1024, default {BATCH_SIZE})"
    )
    parser.add_argument(
        "-L", "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Learning rate (0.0 to 1.0, default {LEARNING_RATE})"
    )
    parser.add_argument(
        "-c", "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help="Where to cache the preprocessed images "
             f"(default: {DEFAULT_CACHE_DIR}). "
             "Ignored with --prepared-data"
    )
    parser.add_argument(
        "-m", "--model-output", type=Path, default=DEFAULT_MODEL_PATH,
        help="Path for writing the best model at the end of the session "
             f"(default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "-z", "--zip-output", type=Path, default=DEFAULT_ZIP_PATH,
        help="Path for writing the zip asked for in the subject pdf "
             f"(default: {DEFAULT_ZIP_PATH})"
    )
    parser.add_argument(
        "-j", "--class-map-output", type=Path,
        default=DEFAULT_CLASS_MAP_PATH,
        help="Path for writing the class-to-index mapping "
             f"(default: {DEFAULT_CLASS_MAP_PATH})"
    )
    parser.add_argument(
        "-i", "--img-dim-output", type=Path, default=DEFAULT_IMG_DIM_PATH,
        help="Path for writing the recorded input image size "
             f"(default: {DEFAULT_IMG_DIM_PATH})"
    )
    args = parser.parse_args()

    if (args.prepared_data is None and args.directory is None):
        parser.error(
            "Either give a directory of raw inputs or pass the "
            "--prepared-data option"
        )
    if (args.prepared_data is not None and args.directory is not None):
        parser.error(
            "Raw inputs directory and --prepared-data are mutually exclusive"
        )
    if (args.device == "cuda" and not torch.cuda.is_available()):
        parser.error("CUDA requested but not available on this machine")
    if (not (0.0 <= args.dropout <= 0.9)):
        parser.error("Valid dropout % range: 0.0 to 0.9")
    if (args.epochs < 1):
        parser.error("Please enter a positive number of epochs instead")
    if (not (16 <= args.input_size <= 512)):
        parser.error("Your input size is quite unreasonable in my opinion")
    if (not (2 <= args.batch_size <= 1024)):
        parser.error("Your batch size is quite unreasonable in my opinion")
    if (not (0.0 <= args.learning_rate <= 1.0)):
        parser.error(
            "The learning rate specified is found to be quite invalid"
        )
    if (args.cache_dir.exists() and not args.prepared_data):
        parser.error(
            f"Cache dir {args.cache_dir} already exists"
        )

    return args


def main():
    print("Welcome to the training program")
    args = parse_args()
    print("[ OK ] args parse")
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"

    if (args.prepared_data is not None):
        print(f"[INFO] prepared data: {args.prepared_data}")
        train_dir = args.prepared_data / "train"
        val_dir = args.prepared_data / "val"
        class_to_idx = discover_prepared_classes(args.prepared_data)
        print("[ OK ] classes discovered")
        train_samples = load_prepared_split(train_dir, class_to_idx)
        print("[ OK ] train split loaded")
        val_samples = load_prepared_split(val_dir, class_to_idx)
        print("[ OK ] val split loaded")
    else:
        print(f"[INFO] starting without pre-prepared data")
        train_cache = args.cache_dir / "train"
        val_cache = args.cache_dir / "val"

        raw_samples, class_to_idx = discover_classes(
            args.directory, max_images_per_class=args.max_images_per_class
        )
        print("[ OK ] classes discovered")

        labels_for_split = [cls for _, cls in raw_samples]
        train_raw, val_raw = train_test_split(
            raw_samples,
            test_size=VAL_SPLIT,
            stratify=labels_for_split,
            random_state=42
        )
        print("[ OK ] split of raw succeeded")

        print(f"[INFO] preparing training data ({len(train_raw)} images)")
        train_samples = prepare_split(
            train_raw,
            class_to_idx,
            train_cache,
            needs_augmenting=True,
            inpsize=args.input_size
        )
        print("[ OK ] training data split prepared")
        print(f"[INFO] preparing valdation data ({len(val_raw)} images)")
        val_samples = prepare_split(
            val_raw,
            class_to_idx,
            val_cache,
            needs_augmenting=False,
            inpsize=args.input_size
        )
        print("[ OK ] validation data split prepared")
        train_dir = train_cache

    train_loader = DataLoader(
        LeafDataset(train_samples, size=args.input_size),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_cuda
    )
    print("[ OK ] training data loader created")
    val_loader = DataLoader(
        LeafDataset(val_samples, size=args.input_size),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_cuda
    )
    print("[ OK ] validation data loader created")

    model = LeafCNN(
        num_classes=len(class_to_idx),
        d_o_p=args.dropout
    ).to(device)
    print("[ OK ] LeafCNN created")
    class_weights = build_class_weights(train_samples, class_to_idx, device)
    print("[ OK ] class weights built")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("[ OK ] criterion built")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print("[ OK ] optimizer built")

    print("[INFO] starting the training\n\n")
    best_val_acc = train(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        device,
        args.epochs,
        PATIENCE,
        args.model_output,
    )
    print(f"best val_acc: {best_val_acc:.4f}")

    package_outputs(
        class_to_idx,
        train_dir,
        args.input_size,
        args.model_output,
        args.zip_output,
        args.class_map_output,
        args.img_dim_output,
    )


if (__name__ == "__main__"):
    main()
