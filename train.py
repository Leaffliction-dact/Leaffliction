import argparse
import json
import random
import shutil
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

# start small for iteration speed try 224/256 later
# btw, 64 apparently works great
INPUT_SIZE = 128
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
DROPOUT_P = 0.3

MAX_EPOCHS = 50
PATIENCE = 7

# for augs
MAX_CLASS_SIZE_MULTIPLIER = 3

# seemingly better for 'realism' and prediction later.
AUGMENT_BEFORE_MASK = True

CACHE_DIR = Path("~/sgoinfre/_prepared_data").expanduser()
TRAIN_CACHE = CACHE_DIR / "train"
VAL_CACHE = CACHE_DIR / "val"
OUT_DIR = Path("~/sgoinfre/leafzip").expanduser()


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


# also use in predict.py
def mask_and_resize(img: np.ndarray, size=INPUT_SIZE) -> np.ndarray:
    saturation = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    mask = pcv.threshold.otsu(gray_img=saturation, object_type="light")
    mask = pcv.fill(bin_img=mask, size=200)
    masked = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    return cv2.resize(masked, (size, size))


def augmented_count_needed(class_count: int, max_count: int) -> int:
    if (class_count >= max_count):
        return 0
    target_count = min(max_count, class_count * MAX_CLASS_SIZE_MULTIPLIER)
    return target_count - class_count


def _process_and_save(
        raw_path: Path,
        out_dir: Path,
        inpsize: int,
        effect_name=None,
        ):
    img = cv2.imread(str(raw_path))

    if (effect_name is None):
        img = mask_and_resize(img, inpsize)
    elif (AUGMENT_BEFORE_MASK):
        img = apply_effects([img], [effect_name], 1)[1]
        img = mask_and_resize(img, inpsize)
    else:
        img = mask_and_resize(img, inpsize)
        img = apply_effects([img], [effect_name], 1)[1]

    tag = effect_name
    if (tag is None):
        tag = "orig"
    out_path = out_dir / f"{raw_path.stem}_{tag}.jpg"
    cv2.imwrite(str(out_path), img)
    return out_path


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
            n_needed = augmented_count_needed(len(raw_paths), max_count)
            combos = [
                (raw_path, name)
                for raw_path in raw_paths
                for name in effect_names
            ]
            chosen = random.sample(combos, min(n_needed, len(combos)))
            for raw_path, name in chosen:
                p = _process_and_save(
                    raw_path,
                    class_dir,
                    effect_name=name,
                    inpsize=inpsize
                )
                prepared.append((p, label))
            print(f"    +{len(chosen)} augmented for {cls}")

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
        patience):
    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

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
                torch.save(model.state_dict(), "best_model.pt")

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


def package_outputs(class_to_idx, train_dir: Path, zip_name="learnings.zip"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_name = OUT_DIR / zip_name
    base_dir = train_dir.parent.parent
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("best_model.pt")

        with open("class_to_idx.json", "w") as f:
            json.dump(class_to_idx, f)
        zf.write("class_to_idx.json")

        for path in train_dir.rglob("*.jpg"):
            zf.write(path, arcname=str(path.relative_to(base_dir)))


def main():
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
             "one side only (since the inputs are squares)."
             f"\nDefault is {INPUT_SIZE}, ranges 16..512"
    )
    parser.add_argument(
        "-B", "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (2..1024, default {BATCH_SIZE})"
    )
    parser.add_argument(
        "-L", "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Learning rate (0.0 to 1.0, default {LEARNING_RATE})"
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
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"

    if (args.prepared_data is not None):
        train_dir = args.prepared_data / "train"
        val_dir = args.prepared_data / "val"
        class_to_idx = discover_prepared_classes(args.prepared_data)
        print(f"using preprepared data from {args.prepared_data}")
        train_samples = load_prepared_split(train_dir, class_to_idx)
        val_samples = load_prepared_split(val_dir, class_to_idx)
    else:
        if (CACHE_DIR.exists()):
            shutil.rmtree(CACHE_DIR)

        raw_samples, class_to_idx = discover_classes(
            args.directory, max_images_per_class=args.max_images_per_class
        )

        labels_for_split = [cls for _, cls in raw_samples]
        train_raw, val_raw = train_test_split(
            raw_samples,
            test_size=VAL_SPLIT,
            stratify=labels_for_split,
            random_state=42
        )

        print(f"preparing training data ({len(train_raw)} images)")
        train_samples = prepare_split(
            train_raw,
            class_to_idx,
            TRAIN_CACHE,
            needs_augmenting=True,
            inpsize=args.input_size
        )
        print(f"preparing valdation data ({len(val_raw)} images)")
        val_samples = prepare_split(
            val_raw,
            class_to_idx,
            VAL_CACHE,
            needs_augmenting=False,
            inpsize=args.input_size
        )
        train_dir = TRAIN_CACHE

    train_loader = DataLoader(
        LeafDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_cuda
    )
    val_loader = DataLoader(
        LeafDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_cuda
    )

    model = LeafCNN(
        num_classes=len(class_to_idx),
        d_o_p=args.dropout
    ).to(device)
    class_weights = build_class_weights(train_samples, class_to_idx, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = train(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        device,
        args.epochs,
        PATIENCE,
    )
    print(f"best val_acc: {best_val_acc:.4f}")

    package_outputs(class_to_idx, train_dir)


if (__name__ == "__main__"):
    main()
