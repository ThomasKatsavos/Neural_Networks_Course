"""
CNN benchmark στο CIFAR-100 υπό label poisoning.

Poisoning Experiments (9 runs συνολικά):
  - baseline (p=0)
  - symmetric  για p∈{0.2, 0.4, 0.6, 0.8}
  - asymmetric για p∈{0.2, 0.4, 0.6, 0.8}
"""
import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from poisoning import PoisonedCIFAR100
from cnn_model import CIFAR100CNN
from cnn_train import train_one_run

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DATA_ROOT     = '../PRELIMINARY/data'
EPOCHS        = 30
BATCH_SIZE    = 128
LR            = 1e-3
WEIGHT_DECAY  = 5e-4
BASE_CHANNELS = 32
NUM_WORKERS   = 4
SEED          = 42

MEAN = (0.5071, 0.4865, 0.4409)
STD  = (0.2673, 0.2564, 0.2762)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Device: {device} "
      f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

# ------------------------------------------------------------------
# Transforms(κατα βάση crop και flip) (augmentation αποκλειστικα στο train)
# ------------------------------------------------------------------
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ------------------------------------------------------------------
# Visualization 
# ------------------------------------------------------------------
def visualize_predictions(model, test_loader, class_names, device,
                          save_path='predictions_grid.png',
                          n_correct=8, n_incorrect=8):
    #Grid με σωστές & λάθος προβλέψεις του baseline CNN.
    model.eval()
    correct_samples, incorrect_samples = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for i in range(x.size(0)):
                if pred[i] == y[i] and len(correct_samples) < n_correct:
                    correct_samples.append((x[i].cpu(), pred[i].item(), y[i].item()))
                elif pred[i] != y[i] and len(incorrect_samples) < n_incorrect:
                    incorrect_samples.append((x[i].cpu(), pred[i].item(), y[i].item()))
            if (len(correct_samples) >= n_correct and
                len(incorrect_samples) >= n_incorrect):
                break

    samples = correct_samples + incorrect_samples
    cols = 4
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    mean = torch.tensor(MEAN).view(3, 1, 1)
    std  = torch.tensor(STD).view(3, 1, 1)

    for i, (img, pred, true) in enumerate(samples):
        img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        color = 'green' if pred == true else 'red'
        axes[i].set_title(
            f"pred: {class_names[pred]}\ntrue: {class_names[true]}",
            color=color, fontsize=8)
        axes[i].axis('off')

    for j in range(len(samples), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved prediction grid to '{save_path}'")


# ------------------------------------------------------------------
# Experiments
# ------------------------------------------------------------------
experiments = [('baseline', 0.0)]
for ptype in ['symmetric', 'asymmetric']:
    for p in [0.2, 0.4, 0.6, 0.8]:
        experiments.append((ptype, p))

print(f"[INFO] Running {len(experiments)} experiments "
      f"× {EPOCHS} epochs | lr={LR}, wd={WEIGHT_DECAY}, "
      f"base_channels={BASE_CHANNELS}")

results = []
baseline_model = None
baseline_test_loader = None
class_names = None

for exp_idx, (ptype, p) in enumerate(experiments, start=1):
    print(f"\n{'='*70}")
    print(f"  [{exp_idx}/{len(experiments)}] "
          f"poison_type={ptype.upper()}  p={p:.2f}")
    print(f"{'='*70}")

    # Reproducibility: ίδια αρχικά βάρη για όλα τα runs
    torch.manual_seed(SEED)

    # PoisonedCIFAR100 χρειάζεται valid poison_type ακόμη και για baseline
    effective_ptype = 'symmetric' if ptype == 'baseline' else ptype

    train_ds = PoisonedCIFAR100(
        root=DATA_ROOT, train=True, p=p,
        poison_type=effective_ptype,
        transform=train_transform, download=False, seed=SEED)

    test_ds = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=False,
        transform=test_transform, download=False)

    if class_names is None:
        class_names = train_ds.base.classes

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    model = CIFAR100CNN(base_channels=BASE_CHANNELS).to(device)
    history = train_one_run(
        model, train_loader, test_loader,
        epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
        optimizer_name='adam', device=device, verbose=True)

    best_acc  = max(history['test_acc'])
    final_acc = history['test_acc'][-1]
    best_ep   = history['epoch'][history['test_acc'].index(best_acc)]

    results.append({
        'poison_type':    ptype,
        'p':              p,
        'best_test_acc':  best_acc,
        'best_epoch':     best_ep,
        'final_test_acc': final_acc,
        'time_s':         history['time'],
    })

    if p == 0.0:
        baseline_model = model
        baseline_test_loader = test_loader

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
print("\n" + "=" * 80)
print(f"{'Type':<12} {'p':>5} {'Best Acc':>10} {'@ep':>5} "
      f"{'Final Acc':>11} {'Time (s)':>10}")
print("-" * 80)
for r in results:
    print(f"{r['poison_type']:<12} {r['p']:>5.2f} "
          f"{r['best_test_acc']:>9.2f}% {r['best_epoch']:>5d} "
          f"{r['final_test_acc']:>10.2f}% {r['time_s']:>9.1f}")
print("=" * 80)

# Relative collapse vs baseline
baseline_acc = next(r for r in results if r['p'] == 0.0)['best_test_acc']
print(f"\nRelative accuracy (% of baseline {baseline_acc:.2f}%):")
for ptype in ['symmetric', 'asymmetric']:
    print(f"\n{ptype.upper()}:")
    for r in results:
        if r['poison_type'] != ptype:
            continue
        rel = r['best_test_acc'] / baseline_acc * 100
        print(f"  p={r['p']:.2f}  →  {rel:5.1f}%  "
              f"(absolute {r['best_test_acc']:.2f}%)")

# ------------------------------------------------------------------
# Visualization grid (baseline model)
# ------------------------------------------------------------------
if baseline_model is not None:
    print("\n[INFO] Generating prediction grid from baseline model...")
    visualize_predictions(baseline_model, baseline_test_loader, class_names, device)

print("\n[INFO] Done.")