# CNN: CIFAR-100 Classification under Label Poisoning

Εκπαίδευση Συνελικτικού Νευρωνικού Δικτύου (CNN) στο CIFAR-100 και μελέτη της ανθεκτικότητάς του υπό συνθήκες αλλοίωσης ετικετών (label poisoning).

Όλα τα πειράματα εκτελούνται σε **NVIDIA A100-SXM4-40GB** (partition `ampere`).

## Δεδομένα

Χρησιμοποιείται το **CIFAR-100**:
- 50.000 εικόνες εκπαίδευσης, 10.000 εικόνες ελέγχου
- 100 κλάσεις, οργανωμένες σε 20 υπερ-κλάσεις (5 fine classes ανά superclass)
- Εικόνες 3 × 32 × 32 (RGB)

## Δομή

```
.
├── poisoning.py          # Poison functions & PoisonedCIFAR100 Dataset
├── cnn_model.py          # VGG-style CNN
├── cnn_train.py          # Training loop με LR scheduler
├── cnn_benchmark.py      # Κύριο label poisoning benchmark
├── ablation_study.py     # Depth / LR / weight decay ablation
├── run_cnn.slurm
└── run_ablation.slurm
```

## Αρχιτεκτονική CNN

Το δίκτυο ακολουθεί το **VGG-style** μοτίβο με επαναλαμβανόμενα blocks:

```
Κάθε block: [Conv2d 3×3 → BatchNorm → ReLU] × 2 → MaxPool2d 2×2
```

Τα κανάλια διπλασιάζονται σε κάθε block (32 → 64 → 128), ενώ το spatial μέγεθος υποδιπλασιάζεται (32 → 16 → 8 → 4). Μετά από 3 blocks, ο tensor `(128, 4, 4)` ισοπεδώνεται σε διάνυσμα 2048 χαρακτηριστικών και περνά από δύο πλήρως συνδεδεμένα επίπεδα:

```
Flatten(2048) → Linear(2048, 512) → ReLU → Dropout(0.5) → Linear(512, 100)
```

Το βάθος (`n_blocks`) και το πλάτος (`base_channels`) είναι παράμετροι, ώστε να υποστηρίζεται ablation study χωρίς αλλαγή κώδικα.

## Label Poisoning

Δύο τύποι αλλοίωσης ετικετών, εφαρμόζονται **μόνο στο training set** (το test set παραμένει πάντα καθαρό):

- **Symmetric:** το `p`% των labels αντικαθίσταται με ένα ομοιόμορφα τυχαίο διαφορετικό label από τις 99 υπόλοιπες κλάσεις.
- **Asymmetric:** το `p`% αντικαθίσταται με ένα τυχαίο διαφορετικό label της **ίδιας** υπερ-κλάσης (semantically coherent noise).

Το poisoning εφαρμόζεται μία φορά κατά την αρχικοποίηση του `PoisonedCIFAR100` και παραμένει σταθερό σε όλα τα epochs.

## Εκτέλεση

```bash
# Κύριο benchmark (9 runs, ~21 min)
python3 cnn_benchmark.py

# Ablation study (10 runs, ~23 min)
python3 ablation_study.py
```

Σε HPC cluster με SLURM:

```bash
sbatch run_cnn.slurm
sbatch run_ablation.slurm
```

Απαιτείται PyTorch με CUDA. Για το prediction grid απαιτείται επίσης `matplotlib`.

## Αποτελέσματα

### Κύριο benchmark (baseline 58.52%)

| Poisoning | p=0.2 | p=0.4 | p=0.6 | p=0.8 |
|---|---|---|---|---|
| Symmetric | 52.82% | 46.43% | 24.08% | 5.62% |
| Asymmetric | 54.17% | 47.76% | 33.62% | 14.68% |

**Κύρια ευρήματα:**

- Το asymmetric poisoning είναι **λιγότερο** καταστροφικό από το symmetric στο CNN. Εξήγηση: το semantic noise είναι συνεπές, άρα το δίκτυο μαθαίνει χρήσιμα features της υπερ-κλάσης ακόμη και από λάθος labels.
- Στο p=0.6 η διαφορά μεταξύ των δύο τύπων είναι 9.5 ποσοστιαίες μονάδες, ενώ στο p=0.8 φτάνει τις 9.1 μονάδες.

### Ablation study

| Depth | p=0.0 | p=0.4 |
|---|---|---|
| 2 blocks | 51.18% | 41.66% |
| **3 blocks (baseline)** | **58.52%** | **46.43%** |
| 4 blocks | 59.48% | 40.23% |

| LR Schedule | p=0.0 | p=0.4 |
|---|---|---|
| constant (baseline) | 58.52% | 46.43% |
| **cosine** | **60.86%** | **49.35%** |
| step | 56.21% | 39.57% |

| Weight Decay | p=0.0 | p=0.4 |
|---|---|---|
| 5e-4 (baseline) | 58.52% | 46.43% |
| 5e-3 | 48.55% | 27.47% |

**Συμπεράσματα ablation:**

- Το βάθος 3 είναι το βέλτιστο trade-off· το βάθος 4 αυξάνει μεν το clean accuracy (59.48% έναντι 58.52%) αλλά μειώνει την ανθεκτικότητα στο poisoning (40.23% έναντι 46.43%).
- Το cosine LR schedule δίνει +2.3% clean accuracy και +2.9% robustness χωρίς κόστος.
- Υπερβολικό weight decay (5e-3) οδηγεί σε underfitting (-10% clean, -19% poisoned) και καταστροφική πτώση robustness.

## Αναφορές

- Simonyan & Zisserman. *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015.
- Ioffe & Szegedy. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML 2015.
- Srivastava et al. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR 2014.
