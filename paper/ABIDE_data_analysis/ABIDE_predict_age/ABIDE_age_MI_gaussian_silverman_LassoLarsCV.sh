#!/bin/bash
#SBATCH --account=def-masd
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-12:00:00
#SBATCH --job-name=age_MI_gaussian_silverman_LassoLarsCV

module load arch/avx2 gcc llvm rust arrow nodejs python/3.10 r/4.0.2 python-build-bundle/2023b

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip Cython

# ### run this block at the login node to build wheels
# module load arch/avx2 gcc llvm rust arrow nodejs python/3.10 r/4.0.2 python-build-bundle/2023b
# ### upgrading the tools
# pip install --upgrade pip setuptools wheel
# ### remove all old wheels
# rm *.whl
# ### get wheels builder
# git clone https://github.com/ComputeCanada/wheels_builder
# export PATH=$PATH:${HOME}/wheels_builder
# ### build KDEpy 1.1.9
# ${HOME}/wheels_builder/unmanylinuxize.sh --package KDEpy --version 1.1.9 --python 3.8,3.9,3.10 --find_links https://files.pythonhosted.org/packages/
# ### built nonconvexAG 1.0.6
# ${HOME}/wheels_builder/unmanylinuxize.sh --package nonconvexAG --version 1.0.6 --python 3.8,3.9,3.10 --find_links https://files.pythonhosted.org/packages/
# ### built dask 2024.2.1
# ${HOME}/wheels_builder/unmanylinuxize.sh --package dask --version 2024.2.1 --python 3.8,3.9,3.10 --find_links https://files.pythonhosted.org/packages/
# ### built fastHDMI 1.25.16
# pip install fastHDMI==1.25.16 --no-cache-dir
# pip wheel fastHDMI --no-deps

# # Here basically to build the packages at login node and install them in slurm job submission locally
pip install --no-index bed-reader numpy sklearn matplotlib scipy numba multiprocess scikit-learn cupy rpy2 pandas dask Cython
pip install --no-index /home/kyang/KDEpy-1.1.9+computecanada-cp310-cp310-linux_x86_64.whl
pip install --no-index /home/kyang/nonconvexAG-1.0.6+computecanada-py3-none-any.whl
pip install --no-index /home/kyang/fastHDMI-1.25.16-cp310-cp310-linux_x86_64.whl
pip install --no-index /home/kyang/dask-2024.2.1+computecanada-py3-none-any.whl

nvidia-smi
lscpu

echo "running ABIDE_age_MI_gaussian_silverman_LassoLarsCV.py"

cp /home/kyang/projects/def-cgreenwo/abide_data/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv $SLURM_TMPDIR/
cp /home/kyang/projects/def-cgreenwo/kyang/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX_CasesOnly.csv $SLURM_TMPDIR/
cp ../ABIDE_columns.npy $SLURM_TMPDIR/
cp ../ABIDE_age_MI_gaussian_silverman_output.npy $SLURM_TMPDIR/

python3 ABIDE_age_MI_gaussian_silverman_LassoLarsCV.py
    