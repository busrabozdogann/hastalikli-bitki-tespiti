"""
Plant Disease Predictor — MobileNetV3-Large
Companion inference script for plant_mobilenet.py

Usage:
    python predict_mobilenet.py --img path/to/leaf.jpg
    python predict_mobilenet.py --img img1.jpg img2.jpg img3.jpg
    python predict_mobilenet.py --folder path/to/folder
"""

import os, argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Large_Weights
from PIL import Image

# ── Config ────────────────────────────────────────────────
ROOT       = "./PlantVillage"
OUT        = "./mobilenet_outputs"
MODEL_PATH = "./mobilenet_outputs/mobilenet_best.pth"
IMAGE_SIZE = 224
DROPOUT    = 0.2
TOP_K      = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ── Class names ───────────────────────────────────────────
base_ds     = datasets.ImageFolder(ROOT)
class_names = base_ds.classes
n_classes   = len(class_names)
print(f"Classes : {n_classes}")

# ── Transform  (same as tf_eval in plant_mobilenet.py) ────
mu  = [0.485, 0.456, 0.406]
sig = [0.229, 0.224, 0.225]

tf_eval = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mu, sig),
])

# ── Model ─────────────────────────────────────────────────
net = models.mobilenet_v3_large(weights=None)

in_dim = net.classifier[0].in_features
net.classifier = nn.Sequential(
    nn.Linear(in_dim, 256),
    nn.Hardswish(),
    nn.Dropout(DROPOUT),
    nn.Linear(256, n_classes),
)

net.load_state_dict(
    torch.load(MODEL_PATH, map_location=device, weights_only=True)
)
net = net.to(device).eval()
print(f"Model  : {MODEL_PATH}\n")

# ── Predict ───────────────────────────────────────────────
def predict(img_path):
    img    = Image.open(img_path).convert("RGB")
    tensor = tf_eval(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(net(tensor), dim=1)[0]
    top_probs, top_idx = probs.topk(TOP_K)
    results = [(class_names[i], p.item()) for i, p in zip(top_idx, top_probs)]
    return img, results

# ── Display ───────────────────────────────────────────────
def show_results(entries):
    n   = len(entries)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1: axes = [axes]

    for row, (img_path, img, preds) in enumerate(entries):
        ax_img, ax_bar = axes[row]

        ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(os.path.basename(img_path), fontsize=9)

        labels = [p[0].replace("_", " ") for p in preds]
        values = [p[1] * 100               for p in preds]
        colors = ["#27ae60" if i == 0 else "#b2bec3" for i in range(len(preds))]

        ax_bar.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.5)
        ax_bar.set_xlim(0, 105)
        ax_bar.set_xlabel("Confidence (%)")
        ax_bar.set_title(f"Top-{TOP_K} Predictions", fontsize=9)
        ax_bar.grid(axis="x", ls="--", alpha=0.4)

        for bar, val in zip(ax_bar.patches, values[::-1]):
            ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=9)

        print(f"📷  {os.path.basename(img_path)}")
        for rank, (cls, prob) in enumerate(preds, 1):
            print(f"  #{rank}  {cls:<50}  {prob*100:.2f}%")
        print()

    plt.suptitle("MobileNetV3-Large — Predictions", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(OUT, "predictions.png")
    plt.savefig(out_path, dpi=150)
    print("Saved", out_path)
    plt.show()

# ── Main ──────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--img",    nargs="+", default=[], help="Path(s) to image file(s)")
parser.add_argument("--folder", default=None,          help="Path to a folder of images")
args = parser.parse_args()

img_paths = list(args.img)
if args.folder:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for f in sorted(os.listdir(args.folder)):
        if os.path.splitext(f)[1].lower() in exts:
            img_paths.append(os.path.join(args.folder, f))

if not img_paths:
    print("No images provided. Use --img or --folder.")
    exit()

print(f"Images : {len(img_paths)}\n")

entries = []
for path in img_paths:
    img, preds = predict(path)
    entries.append((path, img, preds))

show_results(entries)