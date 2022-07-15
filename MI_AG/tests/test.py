import numpy as np
# import cupy as cp
from scipy.linalg import toeplitz, block_diag


# test for LM numpy
np.random.seed(1)
N = 1000
SNR = 5.
true_beta = np.array([2,-2,8,-8]+[0]*1000)
X_cov = toeplitz(.6**np.arange(true_beta.shape[0]))
mean = np.zeros(true_beta.shape[0])
X = np.random.multivariate_normal(mean, X_cov, N)
X -= np.mean(X,0).reshape(1,-1)
X /= np.std(X,0)
intercept_design_column = np.ones(N).reshape(N, 1)
X_sim = np.concatenate((intercept_design_column, X), 1)
true_sigma_sim = np.sqrt(true_beta.T@X_cov@true_beta/SNR)
true_beta_intercept = np.concatenate((np.array([1.23]), true_beta)) # here just define the intercept to be 1.23 for simulated data 
epsilon = np.random.normal(0, true_sigma_sim, N)
y_sim = X_sim@true_beta_intercept + epsilon

lambda_seq = np.arange(40)/400
lambda_seq = lambda_seq[1:]
lambda_seq = lambda_seq[::-1]

# do NOT include the design matrix intercept column 
LM_beta = solution_path_LM_strongrule(design_matrix=X_sim, outcome=y_sim, lambda_=lambda_seq, beta_0 = np.ones(1), tol=1e-2, maxit=500, penalty="SCAD", a=3.7, gamma=2., add_intercept_column=True)

assert LM_beta.dtype() == "float"


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

# assert fit2.dtype() == "float"


# test for logistic numpy
np.random.seed(0)
N = 1000
SNR = 5.
true_beta = np.array([.5,-.5,.8,-.8]+[0]*2000)
X_cov = toeplitz(.5**np.arange(2004))
mean = np.zeros(true_beta.shape[0])
X = np.random.multivariate_normal(mean, X_cov, N)
X -= np.mean(X,0).reshape(1,-1)
X /= np.std(X,0)
intercept_design_column = np.ones(N).reshape(N, 1)
X_sim = np.concatenate((intercept_design_column, X), 1)
true_sigma_sim = np.sqrt(true_beta.T@X_cov@true_beta/SNR)
true_beta_intercept = np.concatenate((np.array([0.5]), true_beta))
signal = X_sim@true_beta_intercept + np.random.normal(0, true_sigma_sim, N)
y_sim = np.random.binomial(1, np.tanh(signal/2)/2+.5)

fit2 = solution_path_logistic(design_matrix=X_sim, outcome=y_sim, tol=1e-2, maxit=500, lambda_=np.linspace(.005,.08,60)[::-1], penalty="SCAD", a=3.7, gamma=2.)

assert fit2.dtype() == "float"


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

# assert fit2.dtype() == "float"

