# Neural Networks – Deep Learning Course

This repository contains the code and project implementations for the
*Neural Networks – Deep Learning* course (Fall semester 2026–27).

The project covers the design, training and evaluation of neural network
classifiers on the CIFAR-100 dataset, along with the scripts required for
execution on the "Aristoteles" HPC cluster (partition `ampere`, NVIDIA A100).

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

**Intermediate assignment**

```
cd PRELIMINARY
python3 compare_classifiers.py
python3 label_poisoning_benchmark.py
```

**First assignment**

```
cd FIRST
python3 cnn_benchmark.py      #  ~20 min
python3 ablation_study.py     #  ~23 min
```

On the Aristoteles cluster, submit the corresponding SLURM scripts:

```
sbatch run_cnn.slurm
sbatch run_ablation.slurm
```

## Author
Thomas Katsavos
