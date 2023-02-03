#!/bin/bash
#SBATCH --account=def-masd
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --time=6-00:00:00
#SBATCH --job-name=ABIDE_age

module load gcc llvm rust arrow cuda nodejs python/3.8.10

virtualenv $HOME/jupyter_py3
source $HOME/jupyter_py3/bin/activate

nvidia-smi
lscpu

python3 ABIDE_age_MI_RidgeCV.py
