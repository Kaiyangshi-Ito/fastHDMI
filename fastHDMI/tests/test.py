import matplotlib.pyplot as plt
from fastHDMI import *
import numpy as np
from bed_reader import open_bed
# import cupy as cp
from scipy.linalg import toeplitz, block_diag
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# test for continuous_screening_parallel on plink files
bed_file = r"./sim/sim1.bed"
bim_file = r"./sim/sim1.bim"
fam_file = r"./sim/sim1.fam"

_bed = open_bed(filepath=bed_file,
                fam_filepath=fam_file,
                bim_filepath=bim_file)
outcome = np.random.rand(_bed.iid_count)
outcome_iid = _bed.iid

true_beta = np.array([4.2, -2.5, 2.6])
for j in np.arange(3):
    outcome += true_beta[j] * _bed.read(np.s_[:, j], dtype=np.int8).flatten()
    print(_bed.read(np.s_[:, j], dtype=np.float64).flatten())

iid_ind = np.random.permutation(np.arange(_bed.iid_count))
outcome = outcome[iid_ind]
outcome_iid = outcome_iid[iid_ind]

MI_continuous = continuous_screening_plink_parallel(bed_file=bed_file,
                                                    bim_file=bim_file,
                                                    fam_file=fam_file,
                                                    outcome=outcome,
                                                    outcome_iid=outcome_iid)

assert np.all(MI_continuous >= 0.)

# testing for plink files screening
bed_file = r"./sim/sim1.bed"
bim_file = r"./sim/sim1.bim"
fam_file = r"./sim/sim1.fam"

_bed = open_bed(filepath=bed_file,
                fam_filepath=fam_file,
                bim_filepath=bim_file)
outcome = np.random.rand(_bed.iid_count)
outcome_iid = _bed.iid

true_beta = np.array([4.2, -2.5, 2.6])
for j in np.arange(3):
    outcome += true_beta[j] * _bed.read(np.s_[:, j], dtype=np.int8).flatten()
    print(_bed.read(np.s_[:, j], dtype=np.float64).flatten())

outcome = np.random.binomial(1, np.tanh(outcome / 2) / 2 + .5)

iid_ind = np.random.permutation(np.arange(_bed.iid_count))
outcome = outcome[iid_ind]
outcome_iid = outcome_iid[iid_ind]

MI_binary = binary_screening_plink_parallel(bed_file=bed_file,
                                            bim_file=bim_file,
                                            fam_file=fam_file,
                                            outcome=outcome,
                                            outcome_iid=outcome_iid)
# starting from 1 because the first left column should be the outcome
assert np.all(MI_binary >= 0.)

# test for clumping for plink files
bed_file = r"./sim/sim1.bed"
bim_file = r"./sim/sim1.bim"
fam_file = r"./sim/sim1.fam"

clump_plink_parallel(bed_file=bed_file,
                     bim_file=bim_file,
                     fam_file=fam_file,
                     num_SNPS_exam=5)

# single-thread continuous version test

a = continuous_screening_csv(r"./sim/sim_continuous.csv")
assert np.all(a >= 0.)

# parallel continuous version test

a = continuous_screening_csv_parallel(r"./sim/sim_continuous.csv")
assert np.all(a >= 0)
b = Pearson_screening_csv_parallel(r"./sim/sim_continuous.csv")
c = continuous_skMI_screening_csv_parallel(r"./sim/sim_continuous.csv")
assert np.all(c >= 0)

# parallel continuous version but using numpy array
csv = pd.read_csv(r"./sim/sim_continuous.csv",
                  encoding='unicode_escape',
                  engine="c")
# here it is because pandas read the first column as the index
X, y = csv.iloc[:, 2:].to_numpy(), csv.iloc[:, 1].to_numpy()

MI = continuous_screening_array_parallel(X, y)
assert np.all(MI >= 0)
skMI = continuous_skMI_array_parallel(X, y, n_neighbors=3)
assert np.all(MI >= 0)

# test for clumping for CSV files
clump_continuous_csv_parallel(csv_file=r"./sim/sim_continuous.csv",
                              num_vars_exam=5)

# single-thread binary version for csv
a = binary_screening_csv(r"./sim/sim_binary.csv")
assert np.all(a >= 0)

# parallel binary version test
a = binary_screening_csv_parallel(r"./sim/sim_binary.csv")
assert np.all(a >= 0)

b = Pearson_screening_csv_parallel(r"./sim/sim_binary.csv")

c = binary_skMI_screening_csv_parallel(r"./sim/sim_binary.csv")
assert np.all(c >= 0)
