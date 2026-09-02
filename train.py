from pathlib import Path
import argparse
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.dataset import discover_class_images
from leafset import LeafDataset
from models import build_model, ARCH_CHOICES
from resnetcnn import IMAGENET_MEAN, IMAGENET_STD

from utils.train_and_image_outs_and_proc import (
    write_outs,
    write_plain_outs,
    prepare_split,
    INPUT_SIZE,
)

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
DROPOUT_P = 0.3

MAX_EPOCHS = 50
PATIENCE = 7

DEFAULT_CACHE_DIR = Path("~/goinfre/_prepared_data").expanduser()
DEFAULT_MODEL_PATH = Path("best_model.pt")
DEFAULT_ZIP_PATH = Path("~/goinfre/leafzip/learnings.zip").expanduser()
DEFAULT_CLASS_MAP_PATH = Path("class_to_idx.json")
DEFAULT_IMG_DIM_PATH = Path("img_dim")

RESNET18_INPUT_SIZE = 224
DEFAULT_MODEL_PATH_RESNET18 = Path("best_model_resnet18.pt")
DEFAULT_CLASS_MAP_PATH_RESNET18 = Path("class_to_idx_resnet18.json")
DEFAULT_IMG_DIM_PATH_RESNET18 = Path("img_dim_resnet18")


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
        "-a", "--arch", choices=ARCH_CHOICES, default="leafcnn",
        help="Model architecture to train (default: leafcnn). "
             "resnet18 uses a pretrained torchvision ResNet18 with a "
             "frozen backbone, and its own default output paths"
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
        "-I", "--input-size", type=int, default=None,
        help="The size to which the inputs will be scaled, in pixels, "
             "one side only (since the inputs are squares). "
             f"Default is {INPUT_SIZE} for leafcnn, "
             f"{RESNET18_INPUT_SIZE} for resnet18. Ranges 16..512."
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
        "-m", "--model-output", type=Path, default=None,
        help="Path for writing the best model at the end of the session "
             f"(default: {DEFAULT_MODEL_PATH} for leafcnn, "
             f"{DEFAULT_MODEL_PATH_RESNET18} for resnet18)"
    )
    parser.add_argument(
        "-z", "--zip-output", type=Path, default=DEFAULT_ZIP_PATH,
        help="Path for writing the zip asked for in the subject pdf "
             f"(default: {DEFAULT_ZIP_PATH}). "
             "Ignored when --arch resnet18 (no zip produced)"
    )
    parser.add_argument(
        "-j", "--class-map-output", type=Path, default=None,
        help="Path for writing the class-to-index mapping "
             f"(default: {DEFAULT_CLASS_MAP_PATH} for leafcnn, "
             f"{DEFAULT_CLASS_MAP_PATH_RESNET18} for resnet18)"
    )
    parser.add_argument(
        "-i", "--img-dim-output", type=Path, default=None,
        help="Path for writing the recorded input image size "
             f"(default: {DEFAULT_IMG_DIM_PATH} for leafcnn, "
             f"{DEFAULT_IMG_DIM_PATH_RESNET18} for resnet18)"
    )
    args = parser.parse_args()

    if (args.input_size is None):
        if (args.arch == "resnet18"):
            args.input_size = RESNET18_INPUT_SIZE
        else:
            args.input_size = INPUT_SIZE

    if (args.model_output is None):
        if (args.arch == "resnet18"):
            args.model_output = DEFAULT_MODEL_PATH_RESNET18
        else:
            args.model_output = DEFAULT_MODEL_PATH

    if (args.class_map_output is None):
        if (args.arch == "resnet18"):
            args.class_map_output = DEFAULT_CLASS_MAP_PATH_RESNET18
        else:
            args.class_map_output = DEFAULT_CLASS_MAP_PATH

    if (args.img_dim_output is None):
        if (args.arch == "resnet18"):
            args.img_dim_output = DEFAULT_IMG_DIM_PATH_RESNET18
        else:
            args.img_dim_output = DEFAULT_IMG_DIM_PATH

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
        print("[INFO] starting without pre-prepared data")
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

    if (args.arch == "resnet18"):
        dataset_mean = IMAGENET_MEAN
        dataset_std = IMAGENET_STD
    else:
        dataset_mean = None
        dataset_std = None

    train_loader = DataLoader(
        LeafDataset(
            train_samples,
            size=args.input_size,
            mean=dataset_mean,
            std=dataset_std
        ),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=use_cuda
    )
    print("[ OK ] training data loader created")
    val_loader = DataLoader(
        LeafDataset(
            val_samples,
            size=args.input_size,
            mean=dataset_mean,
            std=dataset_std
        ),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_cuda
    )
    print("[ OK ] validation data loader created")

    model = build_model(
        args.arch,
        num_classes=len(class_to_idx),
        dropout=args.dropout
    ).to(device)
    print(f"[ OK ] {args.arch} created")
    class_weights = build_class_weights(train_samples, class_to_idx, device)
    print("[ OK ] class weights built")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("[ OK ] criterion built")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.learning_rate)
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

    if (args.arch == "leafcnn"):
        write_outs(
            class_to_idx,
            train_dir,
            args.input_size,
            args.model_output,
            args.zip_output,
            args.class_map_output,
            args.img_dim_output,
        )
    else:
        write_plain_outs(
            class_to_idx,
            args.input_size,
            args.class_map_output,
            args.img_dim_output,
        )


if (__name__ == "__main__"):
    main()
