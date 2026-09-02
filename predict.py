import cv2
import argparse
import json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import torch

from utils.train_and_image_outs_and_proc import mask_and_resize
from models import build_model, ARCH_CHOICES
from resnetcnn import IMAGENET_MEAN, IMAGENET_STD


def predict(model, tensor, device):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        res_probs = torch.softmax(logits, dim=1).squeeze(0)
        print("res_probs...")
        res_list = res_probs.tolist()
        res_idx = res_probs.argmax().item()
        return res_idx, res_probs[res_idx].item(), res_list


MAX_DISPLAY_DIM = 512


def resize_for_display(img, max_dim=MAX_DISPLAY_DIM):
    h, w = img.shape[:2]
    longest_side = max(h, w)
    if (longest_side <= max_dim):
        return img
    else:
        scale = max_dim / longest_side
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def draw(raw, maskd, classname, conf):
    display_raw = resize_for_display(raw)
    fig, axes = plt.subplots(1, 2)
    rgb_raw = cv2.cvtColor(display_raw, cv2.COLOR_BGR2RGB)
    rgb_maskd = cv2.cvtColor(maskd, cv2.COLOR_BGR2RGB)

    axes[0].imshow(rgb_raw)
    axes[0].axis('off')
    axes[1].imshow(rgb_maskd)
    axes[1].axis('off')

    fig.suptitle(
        f"I think it's {classname} with {conf:.1%} confidence",
        fontsize=48,
    )
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model-input", type=Path, required=True,
        help="The path to the model"
    )
    parser.add_argument(
        "-j", "--class-map-input", type=Path, required=True,
        help="The path to class map json"
    )
    parser.add_argument(
        "-i", "--img-dim-input", type=Path, required=True,
        help="The path for the dimension used"
    )
    parser.add_argument(
        "-I", "--input-img", type=Path, required=True,
        help="The image you wish to ID"
    )
    parser.add_argument(
        "-d", "--device", choices=["cpu", "cuda"], default="cpu",
        help="Device to run on (default: cpu)"
    )
    parser.add_argument(
        "-a", "--arch", choices=ARCH_CHOICES, default="leafcnn",
        help="Model architecture to load (default: leafcnn). "
             "Must match the --arch the model was trained with"
    )
    args = parser.parse_args()

    return args


def main():
    print("Welcome to predict")
    args = parse_args()
    device = torch.device(args.device)

    cti = json.loads(args.class_map_input.read_text())
    print("[ OK ] Json loaded")
    idx_to_class = {v: k for k, v in cti.items()}

    model = build_model(args.arch, num_classes=len(idx_to_class), dropout=0)
    print(f"[ OK ] {args.arch} constructed")
    sdict = torch.load(args.model_input, map_location=args.device)
    print("[ OK ] state dict loaded")
    model.load_state_dict(sdict)
    print("[ OK ] state dict loaded into the model")
    model.to(device)
    print(f"[ OK ] model sent to {args.device}")
    model.eval()
    print("[ OK ] eval enabled")

    raw_input = cv2.imread(args.input_img)
    print("[ OK ] raw img read")
    masked_resized = mask_and_resize(
        raw_input,
        int(args.img_dim_input.read_text())
    )
    print("[ OK ] img masked & resized")
    if (args.arch == "resnet18"):
        rgb_masked_resized = cv2.cvtColor(masked_resized, cv2.COLOR_BGR2RGB)
        float_img = rgb_masked_resized.astype(np.float32) / 255.0
        float_img = (float_img - np.array(IMAGENET_MEAN))
        float_img = float_img / np.array(IMAGENET_STD)
    else:
        float_img = masked_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(float_img).permute(2, 0, 1).float()
    print("[ OK ] tensor formed")

    predicted_idx, confidence, res_list = predict(model, tensor, device)
    print("[ OK ] prediction done")

    classname = idx_to_class[predicted_idx]
    print(f"{confidence:.1%} {classname}")
    print("While the others:")
    for classname_gen, prob in zip(idx_to_class.values(), res_list):
        print(f"{classname_gen:30s}: {prob:13.8%}")
    draw(raw_input, masked_resized, classname, confidence)


if (__name__ == "__main__"):
    main()
