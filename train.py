import argparse
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from plantcv import plantcv as pcv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from Augmentation import apply_effects
from utils.dataset import discover_class_images
from utils.effects import EffectName

# start small for iteration speed try 224/256 later
INPUT_SIZE = (128, 128)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
DROPOUT_P = 0.3

MAX_EPOCHS = 50
PATIENCE = 7

# maybe 3?
MAX_AUGMENTS_PER_IMAGE = 4

# seemingly better for 'realism' and prediction later.
AUGMENT_BEFORE_MASK = True

# r.i.p. RX 470
DEVICE = torch.device("cpu")

CACHE_DIR = Path("./_prepared_data")
TRAIN_CACHE = CACHE_DIR / "train"
VAL_CACHE = CACHE_DIR / "val"


def discover_classes(root: Path):
    class_images = discover_class_images(root)
    class_to_idx = {
        name: i for i, name in enumerate(sorted(class_images))
    }

    samples = []
    for class_name, images in class_images.items():
        for path in images:
            samples.append((path, class_name))

    return samples, class_to_idx


# also use in predict.py
def mask_and_resize(img: np.ndarray, size=INPUT_SIZE) -> np.ndarray:
    saturation = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    mask = pcv.threshold.otsu(gray_img=saturation, object_type="light")
    mask = pcv.fill(bin_img=mask, size=200)
    masked = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    return cv2.resize(masked, size)


# each class has a different amt of input images, so augs need to
# vary accordingly to fill up the input amounts to an adequate
# margin. but not too much ofc to avoid overfit
def augments_needed_for_class(class_count: int, max_count: int) -> int:
    if class_count >= max_count:
        return 0
    ratio_needed = max_count / class_count
    n = min(int(round(ratio_needed)) - 1,
            MAX_AUGMENTS_PER_IMAGE,
            len(EffectName))
    return max(n, 0)


def _process_and_save(raw_path: Path, out_dir: Path, effect_name=None):
    img = cv2.imread(str(raw_path))

    if effect_name is None:
        img = mask_and_resize(img)
    elif AUGMENT_BEFORE_MASK:
        img = apply_effects([img], [effect_name], 1)[1]
        img = mask_and_resize(img)
    else:
        img = mask_and_resize(img)
        img = apply_effects([img], [effect_name], 1)[1]

    tag = effect_name
    if tag is None:
        tag = "orig"
    out_path = out_dir / f"{raw_path.stem}_{tag}.png"
    cv2.imwrite(str(out_path), img)
    return out_path


def prepare_split(samples,
        class_to_idx,
        out_dir: Path,
        needs_augmenting: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = []

    if needs_augmenting:
        counts = {}
        for _, cls in samples:
            counts[cls] = counts.get(cls, 0) + 1
        max_count = max(counts.values())

    effect_names = [effect.name for effect in EffectName]

    for raw_path, cls in samples:
        class_dir = out_dir / cls
        class_dir.mkdir(exist_ok=True)
        label = class_to_idx[cls]

        p = _process_and_save(raw_path, class_dir)
        prepared.append((p, label))

        if needs_augmenting:
            n = needs_augmentings_needed_for_class(counts[cls], max_count)
            for name in effect_names[:n]:
                p = _process_and_save(raw_path, class_dir, effect_name=name)
                prepared.append((p, label))

    return prepared


class LeafDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(str(path))
        arr = img.astype(np.float32) / 255.0
        # HWC -> CHW for arch purposes I guess?
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, label


class LeafCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(DROPOUT_P),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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
        images, labels = images.to(device), labels.to(device)
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

    for epoch in range(max_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"no val_loss improvement for {patience} epochs, "
                  "stopping early")
            break

    return best_val_acc


def package_outputs(class_to_idx, zip_path="learnings.zip"):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write("best_model.pt")

        with open("class_to_idx.json", "w") as f:
            json.dump(class_to_idx, f)
        zf.write("class_to_idx.json")

        for path in TRAIN_CACHE.rglob("*.png"):
            zf.write(path, arcname=str(path.relative_to(CACHE_DIR.parent)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)

    raw_samples, class_to_idx = discover_classes(args.directory)

    labels_for_split = [cls for _, cls in raw_samples]
    train_raw, val_raw = train_test_split(
        raw_samples,
        test_size=VAL_SPLIT,
        stratify=labels_for_split,
        random_state=42
    )

    train_samples = prepare_split(
        train_raw,
        class_to_idx,
        TRAIN_CACHE,
        augment=True
    )
    val_samples = prepare_split(
        val_raw,
        class_to_idx,
        VAL_CACHE,
        augment=False
    )

    train_loader = DataLoader(
        LeafDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        LeafDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = LeafCNN(num_classes=len(class_to_idx)).to(DEVICE)
    class_weights = build_class_weights(train_samples, class_to_idx, DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = train(
        train_loader,
        val_loader,
        model,
        criterion,
        optimizer,
        DEVICE,
        MAX_EPOCHS,
        PATIENCE,
    )
    print(f"best val_acc: {best_val_acc:.4f}")

    package_outputs(class_to_idx)


if __name__ == "__main__":
    main()
