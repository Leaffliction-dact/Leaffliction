import cv2
import argparse
import json
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import torch

from train import mask_and_resize
from leafcnn import LeafCNN


def predict(model, tensor, device):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        res_probs = torch.softmax(logits, dim=1).squeeze(0)
        res_idx = res_probs.argmax().item()
        return res_idx, res_probs[res_idx].item()


def draw(raw, maskd, classname, conf):
    fig, axes = plt.subplots(1, 2)
    rgb_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    rgb_maskd = cv2.cvtColor(maskd, cv2.COLOR_BGR2RGB)

    axes[0].imshow(rgb_raw)
    axes[0].axis('off')
    axes[1].imshow(rgb_maskd)
    axes[1].axis('off')

    fig.suptitle(f"I think it's {classname} with {conf:.1%} confidence")
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
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    device = torch.device(args.device)

    cti = json.loads(args.class_map_input.read_text())
    idx_to_class = {v: k for k, v in cti.items()}

    model = LeafCNN(num_classes=len(idx_to_class), d_o_p=0)
    sdict = torch.load(args.model_input, map_location=args.device)
    model.load_state_dict(sdict)
    model.to(device)
    model.eval()

    raw_input = cv2.imread(args.input_img)
    masked_resized = mask_and_resize(
        raw_input,
        int(args.img_dim_input.read_text())
    )
    float_img = masked_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(float_img).permute(2, 0, 1)

    predicted_idx, confidence = predict(model, tensor, device)

    classname = idx_to_class[predicted_idx]
    print(f"{confidence:.1%} {classname}")
    print(f"(the input file's folder was called {args.input_img.parent})")
    draw(raw_input, masked_resized, classname, confidence)


if (__name__=="__main__"):
    main()
