#!/bin/bash
#SBATCH --account=def-cgreenwo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=3-12:00:00
#SBATCH --job-name=poly3_age_MI_triweight_ISJ_MLPRegressor

module load gcc llvm rust arrow cuda nodejs python/3.8.10 r/4.0.2 python-build-bundle

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip Cython

# ### run this block at the login node to build wheels
# module load gcc llvm rust arrow cuda nodejs python/3.8.10 r/4.0.2 python-build-bundle
# ### upgrading the tools
# pip install --upgrade pip setuptools wheel
# ### remove all old wheels
# rm *.whl
# ### get wheels builder
# git clone https://github.com/ComputeCanada/wheels_builder
# export PATH=$PATH:${HOME}/wheels_builder
# ### build KDEpy 1.1.5
# ${HOME}/wheels_builder/unmanylinuxize.sh --package KDEpy --version 1.1.5 --python 3.8,3.9,3.10 --find_links https://files.pythonhosted.org/packages/
# ### built nonconvexAG 1.0.6
# ${HOME}/wheels_builder/unmanylinuxize.sh --package nonconvexAG --version 1.0.6 --python 3.8,3.9,3.10 --find_links https://files.pythonhosted.org/packages/
# ### built fastHDMI 1.23.18
# pip install fastHDMI==1.23.18 --no-cache-dir
# pip wheel fastHDMI --no-deps

# # Here basically to build the packages at login node and install them in slurm job submission locally
pip install --no-index bed-reader numpy sklearn matplotlib scipy numba multiprocess scikit-learn cupy rpy2 pandas dask Cython
pip install --no-index /home/kyang/KDEpy-1.1.5+computecanada-cp38-cp38-linux_x86_64.whl
pip install --no-index /home/kyang/nonconvexAG-1.0.6+computecanada-py3-none-any.whl
pip install --no-index /home/kyang/fastHDMI-1.23.18-cp38-cp38-linux_x86_64.whl

nvidia-smi
lscpu

echo "running ABIDE_poly3_age_MI_triweight_ISJ_MLPRegressor.py"

cp /home/kyang/projects/def-cgreenwo/abide_data/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv $SLURM_TMPDIR/
cp ../ABIDE_columns.npy $SLURM_TMPDIR/
cp ../ABIDE_age_MI_triweight_ISJ_output.npy $SLURM_TMPDIR/

python3 ABIDE_poly3_age_MI_triweight_ISJ_MLPRegressor.py
    