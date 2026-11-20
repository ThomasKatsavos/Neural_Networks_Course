#!/bin/bash
#SBATCH --job-name=cnn_cifar100
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00


module load gcc/13.2.0-i python/3.11 cuda/12.4.0 py-matplotlib/3.8.4-7xlniii
source ../PRELIMINARY/pytorch-env/bin/activate


srun python3 -u cnn_benchmark.py
