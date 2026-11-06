#!/bin/bash
#SBATCH --job-name=pytorch-compare
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:59:00


module load gcc/13.2.0-i python/3.11 cuda/12.4.0
source pytorch-env/bin/activate


srun python3 label_poisoning_benchmark.py
