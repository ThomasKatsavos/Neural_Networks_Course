#!/bin/bash
#SBATCH --job-name=cnn_ablation
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00

module load gcc/13.2.0-i python/3.11 cuda/12.4.0
source ../PRELIMINARY/pytorch-env/bin/activate

echo "cwd: $(pwd)"
echo "Start: $(date)"

srun python3 -u ablation_study.py

echo "End: $(date)"