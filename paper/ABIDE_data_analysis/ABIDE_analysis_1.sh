#!/bin/bash
#SBATCH --account=def-masd
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ABIDE

module load gcc llvm rust arrow cuda nodejs python/3.8.10

virtualenv $HOME/jupyter_py3
source $HOME/jupyter_py3/bin/activate

python ABIDE_analysis_1.py