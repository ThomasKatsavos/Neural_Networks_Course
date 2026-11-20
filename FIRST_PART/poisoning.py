"""
Label-poisoning υλοποιήσεις για το σετ CIFAR-100.

  - symmetric_poison / asymmetric_poison : CPU poison functions
  - PoisonedCIFAR100                     : Dataset με poisoned targets
"""
import os
import pickle

import torch
import torchvision
from torch.utils.data import Dataset

N_CLASSES = 100
N_SUPER   = 20


# ------------------------------------------------------------------
# Superclass tables (fine <-> coarse)
# ------------------------------------------------------------------
def _build_superclass_tables(cifar_root):
    # Διαβάζει το raw pickle του CIFAR-100 και επιστρέφει πίνακες αντιστοίχισης:
    #   fine_to_super : ποια superclass περιέχει κάθε fine class (100 θέσεις)
    #   super_to_fines: ποιες 5 fine classes ανήκουν σε κάθε superclass (20×5)
    #   fine_pos      : θέση κάθε fine class μέσα στη superclass της (0..4)
    # Χρησιμοποιούνται μόνο από το asymmetric poisoning, για να αλλάζει
    # το label σε άλλο label της ιδιας superclass.
    path = os.path.join(cifar_root, 'cifar-100-python', 'train')
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='bytes')

    coarse = torch.tensor(d[b'coarse_labels'], dtype=torch.long)
    fine   = torch.tensor(d[b'fine_labels'],   dtype=torch.long)

    fine_to_super = torch.zeros(N_CLASSES, dtype=torch.long)
    for f in range(N_CLASSES):
        fine_to_super[f] = coarse[fine == f].mode().values

    super_to_fines = torch.zeros(N_SUPER, 5, dtype=torch.long)
    for s in range(N_SUPER):
        super_to_fines[s] = (fine_to_super == s).nonzero(as_tuple=True)[0]

    fine_pos = torch.zeros(N_CLASSES, dtype=torch.long)
    for f in range(N_CLASSES):
        s = fine_to_super[f].item()
        fine_pos[f] = (super_to_fines[s] == f).nonzero(as_tuple=True)[0].item()

    return fine_to_super, super_to_fines, fine_pos


# ------------------------------------------------------------------
# Poison functions (CPU)
# ------------------------------------------------------------------
def symmetric_poison(labels, p, seed=None):
    # Αντικαθιστά το p% των labels με ένα ομοιόμορφα τυχαίο διαφορετικό label
    # από τις 99 υπόλοιπες κλάσεις (symmetric noise: χωρίς σημασιολογία).
    n = labels.numel()
    n_poison = int(n * p)
    poisoned = labels.clone()
    if n_poison == 0:
        return poisoned

    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    idx  = torch.randperm(n, generator=gen)[:n_poison]
    orig = labels[idx]
    rand = torch.randint(0, N_CLASSES - 1, (n_poison,), generator=gen)
    rand = torch.where(rand >= orig, rand + 1, rand)
    poisoned[idx] = rand
    return poisoned


def asymmetric_poison(labels, p, tables, seed=None):
    # Στο p% των labels -> random αλλαγή με διαφορετικό label της ίδιας υπερκλάσσης.
    fine_to_super, super_to_fines, fine_pos = tables
    n = labels.numel()
    n_poison = int(n * p)
    poisoned = labels.clone()
    if n_poison == 0:
        return poisoned

    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    idx  = torch.randperm(n, generator=gen)[:n_poison]
    orig = labels[idx]
    s    = fine_to_super[orig]
    pos  = fine_pos[orig]

    rand_pos = torch.randint(0, 4, (n_poison,), generator=gen)
    rand_pos = torch.where(rand_pos >= pos, rand_pos + 1, rand_pos)
    poisoned[idx] = super_to_fines[s, rand_pos]
    return poisoned


# ------------------------------------------------------------------
# Dataset wrapper
# ------------------------------------------------------------------
class PoisonedCIFAR100(Dataset):
    """
    CIFAR-100 με poisoned training labels.
      - Εφαρμόζει poisoning μια μονο φορά στο __init__ (σταθερό για όλα τα epochs).
      - self.original : ground-truth labels
      - self.poisoned : labels που βλέπει ο classifier
    """
    def __init__(self, root, train, p=0.0, poison_type='symmetric',
                 transform=None, download=True, seed=42):
        self.base = torchvision.datasets.CIFAR100(
            root=root, train=train, download=download, transform=transform)
        self.original = torch.tensor(self.base.targets, dtype=torch.long)
        self.poisoned = self._apply_poison(p, poison_type, seed)

    def _apply_poison(self, p, poison_type, seed):
        if p == 0.0:
            return self.original.clone()
        if poison_type == 'symmetric':
            return symmetric_poison(self.original, p, seed=seed)
        elif poison_type == 'asymmetric':
            tables = _build_superclass_tables(self.base.root)
            return asymmetric_poison(self.original, p, tables, seed=seed)
        else:
            raise ValueError(f"Unknown poison_type: {poison_type}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]     # αγνοούμε το original label
        return img, self.poisoned[idx].item()    # επιστρέφουμε το νέο, πιθανώς poisoned label