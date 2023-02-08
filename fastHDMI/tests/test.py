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

# parallel binary version for csv
a = binary_screening_csv_parallel(r"./sim/sim_binary.csv",
                                  csv_engine="pyarrow")
assert np.all(a >= 0)

# test for LM numpy
np.random.seed(1)
N = 1000
SNR = 5.
true_beta = np.array([2, -2, 8, -8] + [0] * 1000)
X_cov = toeplitz(.6**np.arange(true_beta.shape[0]))
mean = np.zeros(true_beta.shape[0])
X = np.random.multivariate_normal(mean, X_cov, N)
X -= np.mean(X, 0).reshape(1, -1)
X /= np.std(X, 0)
intercept_design_column = np.ones(N).reshape(N, 1)
X_sim = np.concatenate((intercept_design_column, X), 1)
true_sigma_sim = np.sqrt(true_beta.T @ X_cov @ true_beta / SNR)
# here just define the intercept to be 1.23 for simulated data
true_beta_intercept = np.concatenate((np.array([1.23]), true_beta))
epsilon = np.random.normal(0, true_sigma_sim, N)
y_sim = X_sim @ true_beta_intercept + epsilon

lambda_seq = np.arange(40) / 400
lambda_seq = lambda_seq[1:]
lambda_seq = lambda_seq[::-1]

# do NOT include the design matrix intercept column
LM_beta = solution_path_LM_strongrule(design_matrix=X_sim,
                                      outcome=y_sim,
                                      lambda_=lambda_seq,
                                      beta_0=np.ones(1),
                                      tol=1e-2,
                                      maxit=500,
                                      penalty="SCAD",
                                      a=3.7,
                                      gamma=2.,
                                      add_intercept_column=True)

assert LM_beta.dtype == "float"

# single-thread continuous screening for csv
a = continuous_screening_csv(r"./sim/sim_continuous.csv")

# single-thread binary screening for csv
a = binary_screening_csv(r"./sim/sim_binary.csv")

# test for LM cupy
# cp.random.seed(0)
# N = 1000
# p_zeros = 2000
# SNR = 5.
# true_beta = cp.array([2,-2,8,-8]+[0]*p_zeros)
# X_cov = toeplitz(.6**np.arange(true_beta.shape[0]))
# X_cov = cp.asarray(X_cov)
# mean = cp.zeros(len(true_beta))
# X = cp.random.multivariate_normal(mean, X_cov, N)
# X -= cp.mean(X,0).reshape(1,-1)
# X /= cp.std(X,0)
# intercept_design_column = cp.ones(N).reshape(N, 1)
# X_sim = cp.concatenate((intercept_design_column, X), 1)
# true_sigma_sim = cp.sqrt(true_beta.T@X_cov@true_beta/SNR)
# true_beta_intercept = cp.concatenate((cp.array([1.23]), true_beta)) # here just define the intercept to be 1.23 for simulated data
# epsilon = cp.random.normal(0, true_sigma_sim, N)
# y_sim = X_sim@true_beta_intercept + epsilon

# fit2 = solution_path_LM(design_matrix=X_sim, outcome=y_sim, tol=1e-2, maxit=500, lambda_=cp.linspace(.1,1,100), penalty="SCAD", a=3.7, gamma=2.)

# assert fit2.dtype == "float"

# test for logistic numpy
np.random.seed(0)
N = 1000
SNR = 5.
true_beta = np.array([.5, -.5, .8, -.8] + [0] * 2000)
X_cov = toeplitz(.5**np.arange(2004))
mean = np.zeros(true_beta.shape[0])
X = np.random.multivariate_normal(mean, X_cov, N)
X -= np.mean(X, 0).reshape(1, -1)
X /= np.std(X, 0)
intercept_design_column = np.ones(N).reshape(N, 1)
X_sim = np.concatenate((intercept_design_column, X), 1)
true_sigma_sim = np.sqrt(true_beta.T @ X_cov @ true_beta / SNR)
true_beta_intercept = np.concatenate((np.array([0.5]), true_beta))
signal = X_sim @ true_beta_intercept + np.random.normal(0, true_sigma_sim, N)
y_sim = np.random.binomial(1, np.tanh(signal / 2) / 2 + .5)

fit2 = solution_path_logistic(design_matrix=X_sim,
                              outcome=y_sim,
                              tol=1e-2,
                              maxit=500,
                              lambda_=np.linspace(.005, .08, 60)[::-1],
                              penalty="SCAD",
                              a=3.7,
                              gamma=2.)

assert fit2.dtype == "float"

# test for logistic cupy
# cp.random.seed(0)
# N = 1000
# SNR = 5.
# true_beta = cp.array([.5,-.5,.8,-.8]+[0]*2000)
# X_cov = toeplitz(.6**np.arange(true_beta.shape[0]))
# X_cov = cp.asarray(X_cov)
# mean = cp.zeros(true_beta.shape[0])
# X = cp.random.multivariate_normal(mean, X_cov, N)
# X -= cp.mean(X,0).reshape(1,-1)
# X /= cp.std(X,0)
# intercept_design_column = cp.ones(N).reshape(N, 1)
# X_sim = cp.concatenate((intercept_design_column, X), 1)
# true_sigma_sim = cp.sqrt(true_beta.T@X_cov@true_beta/SNR)
# true_beta_intercept = cp.concatenate((cp.array([0.5]), true_beta))
# signal = X_sim@true_beta_intercept + cp.random.normal(0, true_sigma_sim, N)
# y_sim = cp.random.binomial(1, cp.tanh(signal/2)/2+.5)

# fit2 = solution_path_logistic(design_matrix=X_sim, outcome=y_sim, tol=1e-2, maxit=500, lambda_=cp.linspace(.005,.08,60)[::-1], penalty="SCAD", a=3.7, gamma=2.)

# assert fit2.dtype == "float"
