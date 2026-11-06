"""
Poisoning benchmark: symmetric & asymmetric label poisoning στο CIFAR-100.

Μετρά πώς καταρρέει η ακρίβεια των:
  - Nearest Class Centroid(NCC)
  - 1-NN
  - 3-NN

καθώς αυξάνεται το ποσοστό αλλοιωμένων training labels
p->[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
"""
import os
import time
import pickle

import torch
import torchvision
import torchvision.transforms as transforms

from classifiers import (
    dataset_to_tensors,
    precompute_knn,
    knn_accuracy,
    ncc_accuracy,
)


#GPU Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Device: {device} "
      f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()

DATA_ROOT = './data'
N_CLASSES = 100
N_SUPER   = 20


def sync():
    # Συγχρονισμός GPU για ακριβή μέτρηση χρόνου.
    if device == 'cuda':
        torch.cuda.synchronize()


# Φόρτωση CIFAR-100 ως tensors στη GPU (flattened)
transform = transforms.Compose([transforms.ToTensor()])

train_ds = torchvision.datasets.CIFAR100(
    root=DATA_ROOT, train=True,  download=True, transform=transform)
test_ds  = torchvision.datasets.CIFAR100(
    root=DATA_ROOT, train=False, download=True, transform=transform)

print("[INFO] Loading tensors to GPU...")
X_train, y_train_orig = dataset_to_tensors(train_ds, device)   # (50000, 3072), (50000,)
X_test,  y_test       = dataset_to_tensors(test_ds,  device)   # (10000, 3072), (10000,)
print(f"[INFO] X_train: {tuple(X_train.shape)} | X_test: {tuple(X_test.shape)}")
print(f"[INFO] Pixel range: [{X_train.min().item():.2f}, {X_train.max().item():.2f}]")


#Προεπεξεργασια για asymmetric poisoning
def load_coarse_labels(root, split='train'):
    # Διαβάζει τα δεδομένα του CIFAR-100 και επιστρέφει
    #τα coarse labels (τα ονοματα των υπερκλάσσεων) για το δεδομένο split.
    
    path = os.path.join(root, 'cifar-100-python', split)
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
    return torch.tensor(d[b'coarse_labels'], dtype=torch.long)


print("[INFO] Building superclass tables...")
coarse_train = load_coarse_labels(DATA_ROOT, 'train').to(device)   # (50000,)

# fine_to_super[f] = ποια superclass περιέχει τη fine class f 
fine_to_super = torch.zeros(N_CLASSES, dtype=torch.long, device=device)
for f in range(N_CLASSES):
    fine_to_super[f] = coarse_train[y_train_orig == f].mode().values

# super_to_fines[s] = οι 5 fine κλάσεις της superclass s 
super_to_fines = torch.zeros(N_SUPER, 5, dtype=torch.long, device=device)
for s in range(N_SUPER):
    super_to_fines[s] = (fine_to_super == s).nonzero(as_tuple=True)[0]

# fine_pos[f] = θέση της fine class f μέσα στο super_to_fines[s] 
fine_pos = torch.zeros(N_CLASSES, dtype=torch.long, device=device)
for f in range(N_CLASSES):
    s = fine_to_super[f].item()
    fine_pos[f] = (super_to_fines[s] == f).nonzero(as_tuple=True)[0].item()

assert super_to_fines.shape == (N_SUPER, 5), \
    f"Expected (20,5) super_to_fines, got {super_to_fines.shape}"
assert (fine_to_super >= 0).all() and (fine_to_super < N_SUPER).all()
print("[INFO] Superclass tables ready")


# Συναρτήσεις poisoning
def symmetric_poison(labels, p):
    #Symmetric label noise:
    #Επιλέγει τυχαία int(N*p) δείγματα και αντικαθιστά το label τους
    #με ένα ομοιόμορφα τυχαίο label από τις 99 υπόλοιπες κλάσεις.
    
    n = labels.numel()
    n_poison = int(n * p)
    poisoned = labels.clone()
    if n_poison == 0:
        return poisoned

    idx  = torch.randperm(n, device=labels.device)[:n_poison]
    orig = labels[idx]                                                     # (n_poison,)

    # rand στο [0, N-2] = [0, 98]. Αν rand μεγαλύτερο/ισο του αρχικού, μετατόπιση +1 -> [0,99]
    rand = torch.randint(0, N_CLASSES - 1, (n_poison,), device=labels.device)
    rand = torch.where(rand >= orig, rand + 1, rand)

    poisoned[idx] = rand
    return poisoned


def asymmetric_poison(labels, p):
    #Asymmetric label noise:
    #Επιλέγει τυχαία int(N*p) δείγματα και αντικαθιστά το label τους
    #με ένα τυχαίο διαφορετικό label της ίδιας superclass (20 superclasses × 5 classes).
    
    n = labels.numel()
    n_poison = int(n * p)
    poisoned = labels.clone()
    if n_poison == 0:
        return poisoned

    idx  = torch.randperm(n, device=labels.device)[:n_poison]
    orig = labels[idx]
    s    = fine_to_super[orig]                                             #(n_poison,)
    pos  = fine_pos[orig]                                                  #(n_poison,)

    # rand_pos στο [0,3] - ίδιο τέχνασμα με μετατόπιση +1.
    rand_pos = torch.randint(0, 4, (n_poison,), device=labels.device)
    rand_pos = torch.where(rand_pos >= pos, rand_pos + 1, rand_pos)

    poisoned[idx] = super_to_fines[s, rand_pos]
    return poisoned



#Precompute 3-NN indices μια φορά.

print("\n Precomputing 3-NN indices...")
sync(); t0 = time.time()
top3_idx, _ = precompute_knn(X_test, X_train, k=3, chunk_size=500)
sync(); t_pre = time.time() - t0
print(f"[INFO] Done in {t_pre:.2f}s | top3_idx: {tuple(top3_idx.shape)}")


#Benchmark loop

thresholds   = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
poison_types = ['symmetric', 'asymmetric']

results = []
t_loop = time.time()

for ptype in poison_types:
    print(f"\n{'='*65}")
    print(f"  Poisoning: {ptype.upper()}")
    print(f"{'='*65}")
    print(f"{'p':>5} | {'NCC':>8} | {'1-NN':>8} | {'3-NN':>8} | {'time':>6}")
    print("-" * 65)

    for p in thresholds:
        t_row = time.time()
        #  Παραγωγή poisoned labels για αυτό το (ptype, p) 
        if p == 0.0:
            poisoned = y_train_orig.clone()                #clean baseline
        elif ptype == 'symmetric':
            poisoned = symmetric_poison(y_train_orig, p)
        else:
            poisoned = asymmetric_poison(y_train_orig, p)

        # Αξιολόγηση των τριών classifiers στα poisoned labels 
        acc_ncc = ncc_accuracy(X_train, poisoned, X_test, y_test)
        acc_1nn = knn_accuracy(top3_idx, poisoned, y_test, k=1)
        acc_3nn = knn_accuracy(top3_idx, poisoned, y_test, k=3)

        results.append({
            'poison_type': ptype,
            'p':           p,
            'ncc_acc':     acc_ncc,
            '1nn_acc':     acc_1nn,
            '3nn_acc':     acc_3nn,
        })

        print(f"{p:>5.2f} | {acc_ncc:>7.2f}% | {acc_1nn:>7.2f}% | "
              f"{acc_3nn:>7.2f}% | {time.time()-t_row:>5.2f}s")

sync()
print(f"\nBenchmark loop: {time.time()-t_loop:.2f}s total")


# Summary - σχεττική ακρίβεια
# Baseline (p=0) - ίδιο και για τα δύο poison types

baseline = next(r for r in results if r['p'] == 0.0)

print("\n" + "=" * 70)
print(f"Relative accuracy (% of p=0 baseline: "
      f"NCC={baseline['ncc_acc']:.2f}%, "
      f"1NN={baseline['1nn_acc']:.2f}%, "
      f"3NN={baseline['3nn_acc']:.2f}%)")
print("=" * 70)

for ptype in poison_types:
    print(f"\n{ptype.upper()}:")
    print(f"{'p':>5} | {'NCC %':>7} | {'1NN %':>7} | {'3NN %':>7}")
    print("-" * 40)
    for r in results:
        if r['poison_type'] != ptype:
            continue
        ncc_pct = r['ncc_acc'] / baseline['ncc_acc'] * 100
        nn1_pct = r['1nn_acc'] / baseline['1nn_acc'] * 100
        nn3_pct = r['3nn_acc'] / baseline['3nn_acc'] * 100
        print(f"{r['p']:>5.2f} | {ncc_pct:>6.1f}% | {nn1_pct:>6.1f}% | {nn3_pct:>6.1f}%")

print("\n[INFO] Done.")
