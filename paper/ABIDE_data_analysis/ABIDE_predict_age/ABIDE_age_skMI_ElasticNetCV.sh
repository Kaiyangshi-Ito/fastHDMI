
#!/bin/bash
#SBATCH --account=def-cgreenwo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=80G
#SBATCH --time=6-00:00:00
#SBATCH --job-name=skMI_ElasticNetCV

module load gcc llvm rust arrow cuda nodejs python/3.8.10

virtualenv $HOME/jupyter_py3
source $HOME/jupyter_py3/bin/activate

nvidia-smi
lscpu

python3 ABIDE_age_skMI_ElasticNetCV.py
    