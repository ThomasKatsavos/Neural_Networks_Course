#!/bin/bash

# Load necessary modules
module load gcc/13.2.0-i python/3.11

# Create the virtual environment
python3 -m venv pytorch-env

# Activate the environment
source pytorch-env/bin/activate

# Upgrade pip and install PyTorch
pip install --upgrade pip
pip install torch torchvision torchaudio

echo "Environment pytorch-env is set up and activated."
