# Neural Networks – Deep Learning Course

This repository contains the code and project implementations for the
*Neural Networks – Deep Learning* course (Fall semester 2026–27).

The project covers the design, training and evaluation of neural network
classifiers on the CIFAR-100 dataset, along with the scripts required for
execution on the "Aristotelis" HPC cluster (partition `ampere`, NVIDIA A100).

## Repository structure

```
.
├── PRELIMINARY/    # Intermediate assignment: 1-NN, 3-NN, Nearest Class Centroid
└── FIRST/          # First assignment: CNN + label poisoning + ablation study
```

Each subdirectory corresponds to a distinct stage of the project and is
self-contained (own scripts, own SLURM job files).

## Dataset

**CIFAR-100** — 50,000 training images and 10,000 test images, 100 classes
(3 × 32 × 32 RGB).

## Requirements

- Python 3.11
- PyTorch (with CUDA support)
- torchvision, matplotlib (for prediction grids)
- SLURM (for cluster execution)

## Execution

###  *** First follow steps 1-3 from 'PRELIMINARY' folder! ***



On the Aristotelis cluster, submit the corresponding SLURM scripts from PRELIMINARY or FIRST_PART folders, e.g.:

```
sbatch run.sbatch
sbatch run_ablation.sbatch
```

## Author
Thomas Katsavos
