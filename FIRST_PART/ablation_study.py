"""
Ablation study ως προς depth, τύπο LR schedule και weight decay όπως προτάθηκε στην τάξη (10 διαφορετικικά runs).
Baseline: n_blocks=3, constant, wd=5e-4 (από το main benchmark)
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from poisoning import PoisonedCIFAR100
from cnn_model import CIFAR100CNN
from cnn_train import train_one_run

# αρχικοποίηση του πειράματος
DATA_ROOT, EPOCHS, BATCH, BASE_C, WORKERS, SEED = '../PRELIMINARY/data', 30, 128, 32, 4, 42
MEAN, STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

# Ορισμός των παραμέτρων για διενέργεια πειράματος ablation(label, n_blocks, weight_decay, scheduler, p)
EXPERIMENTS = [
    ('depth2_p0', 2, 5e-4, 'constant', 0.0), ('depth2_p4', 2, 5e-4, 'constant', 0.4),
    ('depth4_p0', 4, 5e-4, 'constant', 0.0), ('depth4_p4', 4, 5e-4, 'constant', 0.4),
    ('cosine_p0', 3, 5e-4, 'cosine',   0.0), ('cosine_p4', 3, 5e-4, 'cosine',   0.4),
    ('step_p0',   3, 5e-4, 'step',     0.0), ('step_p4',   3, 5e-4, 'step',     0.4),
    ('wd5e3_p0',  3, 5e-3, 'constant', 0.0), ('wd5e3_p4',  3, 5e-3, 'constant', 0.4),
]

print(f"[INFO] Device: {device} | {len(EXPERIMENTS)} runs × {EPOCHS} epochs")
results = {}

# Διενέργεια όλων των εκτελέσεων, περιλαμβάνεται και poisoning(p=0.4) για σύγκριση με το ground-truth dataset(p=0.0).
for i, (label, nblk, wd, sched, p) in enumerate(EXPERIMENTS, 1):
    print(f"\n[{i}/{len(EXPERIMENTS)}] {label}  "
          f"(n_blocks={nblk}, wd={wd}, sched={sched}, p={p})")
    torch.manual_seed(SEED)
    tr_ds = PoisonedCIFAR100(root=DATA_ROOT, train=True, p=p, poison_type='symmetric',
                             transform=train_tf, download=False, seed=SEED)
    te_ds = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False,
                                          transform=test_tf, download=False)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                           num_workers=WORKERS, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=BATCH, shuffle=False,
                           num_workers=WORKERS, pin_memory=True)

    model = CIFAR100CNN(n_blocks=nblk, base_channels=BASE_C).to(device)
    hist = train_one_run(model, tr_loader, te_loader, epochs=EPOCHS, lr=1e-3,
                         weight_decay=wd, scheduler_name=sched,
                         device=device, verbose=False)
    best = max(hist['test_acc'])
    results[label] = (best, hist['epoch'][hist['test_acc'].index(best)], hist['time'])
    print(f"    → best {best:.2f}% @ ep {results[label][1]}  ({hist['time']:.0f}s)")

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()


def table(title, rows):
    print(f"\n{'='*44}\n{title}\n{'='*44}")
    print(f"{'':<12}{'p=0.0':>14}{'p=0.4':>14}")
    for name, v0, v4 in rows:
        print(f"{name:<12}{v0:>14}{v4:>14}")

# Εκτύπωση αποτελεσμάτων και βέλτιστων παραμέτρων ανα κατηγορία
def v(label, p):
    return f"{results[f'{label}_p{p}'][0]:.2f}%"


table("DEPTH", [
    ("2",   v('depth2', 0), v('depth2', 4)),
    ("3 ***", "BASELINE",     "BASELINE"),
    ("4",   v('depth4', 0), v('depth4', 4)),
])
table("LR SCHEDULE", [
    ("constant ***", "BASELINE",    "BASELINE"),
    ("cosine",     v('cosine', 0), v('cosine', 4)),
    ("step",       v('step', 0),   v('step', 4)),
])
table("WEIGHT DECAY", [
    ("5e-4 ***", "BASELINE",   "BASELINE"),
    ("5e-3",   v('wd5e3', 0), v('wd5e3', 4)),
])
print("\n[INFO] Done.")