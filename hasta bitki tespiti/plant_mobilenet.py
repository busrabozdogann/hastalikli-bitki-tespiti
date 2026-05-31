"""
Plant Disease Detection — MobileNetV3-Large + Transfer Learning
Dataset : PlantVillage (https://www.kaggle.com/datasets/emmarex/plantdisease)

Install:
    pip install torch torchvision scikit-learn matplotlib seaborn tqdm

Usage:
    python plant_mobilenet.py
"""

import os, time, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Large_Weights

from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────
ROOT        = "./PlantVillage"
OUT         = "./mobilenet_outputs"
SEED        = 21
IMAGE_SIZE  = 224
BATCH_SIZE  = 32
EPOCHS      = 12
LR          = 3e-4
DROPOUT     = 0.2
SPLIT       = (0.75, 0.15, 0.10)   # train / val / test

os.makedirs(OUT, exist_ok=True)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ── Transforms ────────────────────────────────────────────
mu  = [0.485, 0.456, 0.406]
sig = [0.229, 0.224, 0.225]

tf_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ColorJitter(hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mu, sig),
])

tf_eval = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mu, sig),
])

# ── Dataset ───────────────────────────────────────────────
base_ds     = datasets.ImageFolder(ROOT)
class_names = base_ds.classes
n_classes   = len(class_names)
n           = len(base_ds)

n_train = int(n * SPLIT[0])
n_val   = int(n * SPLIT[1])
n_test  = n - n_train - n_val

g = torch.Generator().manual_seed(SEED)
train_raw, val_raw, test_raw = random_split(base_ds, [n_train, n_val, n_test], generator=g)

# Apply transforms per split
class Wrapper(torch.utils.data.Dataset):
    def __init__(self, ds, tf): self.ds, self.tf = ds, tf
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        img, lbl = self.ds[i]
        return self.tf(img), lbl

train_ds = Wrapper(train_raw, tf_train)
val_ds   = Wrapper(val_raw,   tf_eval)
test_ds  = Wrapper(test_raw,  tf_eval)

kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
train_dl = DataLoader(train_ds, shuffle=True,  **kw)
val_dl   = DataLoader(val_ds,   shuffle=False, **kw)
test_dl  = DataLoader(test_ds,  shuffle=False, **kw)

print(f"Loaded {n} images | {n_classes} classes")
print(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

# ── Model ─────────────────────────────────────────────────
net = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# Freeze backbone
for p in net.parameters():
    p.requires_grad = False

# Replace classifier
in_dim = net.classifier[0].in_features
net.classifier = nn.Sequential(
    nn.Linear(in_dim, 256),
    nn.Hardswish(),
    nn.Dropout(DROPOUT),
    nn.Linear(256, n_classes),
)

net = net.to(device)
print(f"\nModel : MobileNetV3-Large  |  Head output : {n_classes}")

# ── Optimizer & Scheduler ─────────────────────────────────
# Unfreeze the last conv block + classifier
for name, p in net.named_parameters():
    if "features.16" in name or "classifier" in name:
        p.requires_grad = True

optimizer = optim.RMSprop(
    filter(lambda p: p.requires_grad, net.parameters()),
    lr=LR, alpha=0.99, eps=1e-8, weight_decay=1e-4
)

# Reduce LR when val_loss plateaus — different from StepLR & CosineAnnealing
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.3, patience=2, verbose=True
)

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# ── Helpers ───────────────────────────────────────────────
def step(loader, train=True):
    net.train() if train else net.eval()
    tot_loss, tot_correct = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            if train: optimizer.zero_grad()
            logits = net(x)
            loss   = loss_fn(logits, y)
            if train: loss.backward(); optimizer.step()
            tot_loss    += loss.item() * x.size(0)
            tot_correct += (logits.argmax(1) == y).sum().item()
    return tot_loss / len(loader.dataset), tot_correct / len(loader.dataset)

# ── Training loop ─────────────────────────────────────────
log = dict(tl=[], ta=[], vl=[], va=[])
best_acc, best_state = 0.0, None

print("\n── Training ─────────────────────────────────────")
for ep in range(1, EPOCHS + 1):
    t0 = time.time()
    tl, ta = step(train_dl, train=True)
    vl, va = step(val_dl,   train=False)
    scheduler.step(vl)                      # plateau scheduler uses val loss

    log["tl"].append(tl); log["ta"].append(ta)
    log["vl"].append(vl); log["va"].append(va)

    flag = ""
    if va > best_acc:
        best_acc   = va
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
        torch.save(best_state, os.path.join(OUT, "mobilenet_best.pth"))
        flag = "  ← best"

    print(f"[{ep:02d}/{EPOCHS}]  "
          f"train  loss={tl:.4f}  acc={ta:.4f}  |  "
          f"val  loss={vl:.4f}  acc={va:.4f}  "
          f"({time.time()-t0:.0f}s){flag}")

print(f"\nBest val accuracy : {best_acc:.4f}")

# ── Test evaluation ───────────────────────────────────────
net.load_state_dict(best_state)
net.eval()

preds_all, true_all = [], []
with torch.no_grad():
    for x, y in tqdm(test_dl, desc="Testing"):
        p = net(x.to(device)).argmax(1).cpu().numpy()
        preds_all.extend(p); true_all.extend(y.numpy())

test_acc = np.mean(np.array(preds_all) == np.array(true_all))
print(f"Test accuracy : {test_acc:.4f}\n")
print(classification_report(true_all, preds_all, target_names=class_names, digits=4))

# ── Plots ─────────────────────────────────────────────────
eps = range(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(eps, log["tl"], "b-o", ms=4, label="Train")
axes[0].plot(eps, log["vl"], "r-o", ms=4, label="Val")
axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(ls="--", alpha=0.4)

axes[1].plot(eps, log["ta"], "b-o", ms=4, label="Train")
axes[1].plot(eps, log["va"], "r-o", ms=4, label="Val")
axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(ls="--", alpha=0.4)

plt.suptitle("MobileNetV3-Large — PlantVillage", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=150)
plt.show()

# Confusion matrix
cm = confusion_matrix(true_all, preds_all, normalize="true")
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.3, ax=ax)
ax.set_title("Normalised Confusion Matrix — Test Set")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
plt.show()

print("Done. Outputs saved to:", OUT)