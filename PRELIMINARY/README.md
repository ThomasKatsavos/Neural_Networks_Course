## Introduction
The goal of this project is to compare three simple classifiers, 1-NN, 3-NN and NCC, in a common dataset and use the results as a reference/baseline for further development of more complex structures. It is also a first attempt to integrate a Label-poisoning framework in the training set and measure its effects on the performance.

## Repository structure and contents
The project repository contains:
- log-files(.log) for that recorded some outputs from the cluster
- Three Python source files for the main implementation and tests
- Slurm-script(.sbatch) files for submitting each program as a slurm job on the cluster
- Batch-script files(.sh) used to setup the environment, install dependencies and avtivate a Python venv

## Instructions for exectution
The instructions cover the execution of the code on Aristotelis cluster only. 
1.  Clone the current Github repo on your directory on the university cluster:
``` git clone https://github.com/ThomasKatsavos/Neural_Networks_Course```
You can also just update it in case it is already cloned.

2. After changing the cwd to 'PRELIMINARY', the environement should be setup, installing Pytorch and relative needed moodules, with the command:
   ``` source stup.sh ```.  In case the venv must be activated for modifying files or working on the codebase, use the command: ``` run_env.sh ```.

4. Then, make sure the 'data' folder(the CIFAR100 dataset acquired using Pytorch) lies in the directory, otherwise it will be installed during automatically execution.
5. * For testing the basic classifiers comparison program, submit the correct slurm job with: ``` sbatch run.sbatch ```
   * For testing the Label-poisoning analysis, submit the correct slurm job with: ``` sbatch run_lp.sbatch ```

6. The outputs(log-files) are then created in the cwd after the execution is finished.
