#!/usr/bin/env python
# coding: utf-8

# # Set up the class fundementals

# In[1]:

# import os, sys
# import collections
# import numpy as np
import cupy as cp

# import matplotlib.markers as markers
# import matplotlib.pyplot as plt
# import timeit
# import collections
# from scipy.linalg import toeplitz, block_diag
# from scipy.stats import median_abs_deviation as mad
# import multiprocessing
# import cProfile
# import itertools
# from numba import cuda, vectorize
# import warnings
# warnings.filterwarnings('ignore') # this is just to hide all the warnings
# import rpy2.robjects as robjects
# import matplotlib.pyplot as plt # change font globally to Times
# plt.style.use('ggplot')
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Times New Roman",
#     "font.sans-serif": ["Times New Roman"],
#     "font.size": 12})

# os.chdir(sys.path[0]) # ensure working direcotry is set same as the file

# In[2]:


######################################  some SCAD and MCP things  #######################################
def soft_thresholding(x, lambda_):
    '''
    To calculate soft-thresholding mapping of a given ONE-DIMENSIONAL tensor, BESIDES THE FIRST TERM (so beta_0 will not be penalized). 
    This function is to be used for calculation involving L1 penalty term later. 
    '''
    return cp.hstack((cp.array([x[0]]),
                      cp.where(
                          cp.abs(x[1:]) > lambda_,
                          x[1:] - cp.sign(x[1:]) * lambda_, 0)))


def SCAD(x, lambda_, a=3.7):
    '''
    To calculate SCAD penalty value;
    #x can be a mult# import os, sys
# import collections
import numpy as np
import cupy as cp
import matplotlib.markers as markers
import matplotlib.pyplot as plt
import timeit
# import collections
from scipy.linalg import toeplitz, block_diag
from scipy.stats import median_abs_deviation as mad
# import multiprocessing
# import cProfile
# import itertools
from numba import jit, njit, cuda, vectorize
import warnings
warnings.filterwarnings('ignore') # this is just to hide all the warnings
# import rpy2.robjects as robjects
# import matplotlib.pyplot as plt # change font globally to Times 
# plt.style.use('ggplot')
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Times New Roman",
#     "font.sans-serif": ["Times New Roman"],
#     "font.size": 12})

# os.chdir(sys.path[0]) # ensure working direcotry is set same as the filei-dimensional tensor;
    lambda_, a are scalars;
    Fan and Li suggests to take a as 3.7 
    '''
    # here I notice the function is de facto a function of absolute value of x, therefore take absolute value first to simplify calculation
    x = cp.abs(x)
    temp = cp.where(
        x <= lambda_, lambda_ * x,
        cp.where(x < a * lambda_,
                 (2 * a * lambda_ * x - x**2 - lambda_**2) / (2 * (a - 1)),
                 lambda_**2 * (a + 1) / 2))
    temp[0] = 0.  # this is to NOT penalize intercept beta later
    return temp


def SCAD_grad(x, lambda_, a=3.7):
    '''
    To calculate the gradient of SCAD wrt. icput x; 
    #x can be a multi-dimensional tensor. 
    '''
    # here decompose x to sign and its absolute value for easier calculation
    sgn = cp.sign(x)
    x = cp.abs(x)
    temp = cp.where(
        x <= lambda_, lambda_ * sgn,
        cp.where(x < a * lambda_, (a * lambda_ * sgn - sgn * x) / (a - 1), 0))
    temp[0] = 0.  # this is to NOT penalize intercept beta later
    return temp


def MCP(x, lambda_, gamma):
    '''
    To calculate MCP penalty value; 
    #x can be a multi-dimensional tensor. 
    '''
    # the function is a function of absolute value of x
    x = cp.abs(x)
    temp = cp.where(x <= gamma * lambda_, lambda_ * x - x**2 / (2 * gamma),
                    .5 * gamma * lambda_**2)
    temp[0] = 0.  # this is to NOT penalize intercept beta later
    return temp


def MCP_grad(x, lambda_, gamma):
    '''
    To calculate MCP gradient wrt. icput x; 
    #x can be a multi-dimensional tensor. 
    '''
    temp = cp.where(
        cp.abs(x) < gamma * lambda_,
        lambda_ * cp.sign(x) - x / gamma, cp.zeros_like(x))
    temp[0] = 0.  # this is to NOT penalize intercept beta later
    return temp


def SCAD_concave(x, lambda_, a=3.7):
    '''
    The value of concave part of SCAD penalty; 
    #x can be a multi-dimensional tensor. 
    '''
    x = cp.abs(x)
    temp = cp.where(
        x <= lambda_, 0.,
        cp.where(x < a * lambda_,
                 (lambda_ * x - (x**2 + lambda_**2) / 2) / (a - 1),
                 (a + 1) / 2 * lambda_**2 - lambda_ * x))
    temp[0] = 0.  # this is to NOT penalize intercept beta later
    return temp


def SCAD_concave_grad(x, lambda_, a=3.7):
    '''
    The gradient of concave part of SCAD penalty wrt. icput x; 
    #x can be a multi-dimensional tensor. 
    '''
    sgn = cp.sign(x)
    x = cp.abs(x)
    temp = cp.where(
        x <= lambda_, 0.,
        cp.where(x < a * lambda_, (lambda_ * sgn - sgn * x) / (a - 1),
                 -lambda_ * sgn))
    temp[0] = 0.  # this is to NOT penalize intercept beta later
    return temp


def MCP_concave(x, lambda_, gamma):
    '''
    The value of concave part of MCP penalty; 
    #x can be a multi-dimensional tensor. 
    '''
    # similiar as in MCP
    x = cp.abs(x)
    temp = cp.where(x <= gamma * lambda_, -(x**2) / (2 * gamma),
                    (gamma * lambda_**2) / 2 - lambda_ * x)
    temp[0] = 0.  # this is to NOT penalize intercept beta later
    return temp


def MCP_concave_grad(x, lambda_, gamma):
    '''
    The gradient of concave part of MCP penalty wrt. icput x; 
    #x can be a multi-dimensional tensor. 
    '''
    temp = cp.where(
        cp.abs(x) < gamma * lambda_, -x / gamma, -lambda_ * cp.sign(x))
    temp[0] = 0.  # this is to NOT penalize intercept beta later
    return temp


# # Implementation

# In[3]:


def update_smooth_grad_convex_LM(N, X, beta_md, y):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    return 1. / N * X.T @ (X @ beta_md - y)


def update_smooth_grad_SCAD_LM(N, X, beta_md, y, _lambda, a):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return update_smooth_grad_convex_LM(N=N, X=X, beta_md=beta_md,
                                        y=y) + SCAD_concave_grad(
                                            x=beta_md, lambda_=_lambda, a=a)


def update_smooth_grad_MCP_LM(N, X, beta_md, y, _lambda, gamma):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return update_smooth_grad_convex_LM(
        N=N, X=X, beta_md=beta_md, y=y) + MCP_concave_grad(
            x=beta_md, lambda_=_lambda, gamma=gamma)


def eval_obj_SCAD_LM(N, X, beta_md, y, _lambda, a, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X @ x_temp
    return (error.T @ error) / (2. * N) + cp.sum(
        SCAD(x_temp, lambda_=_lambda, a=a))


def eval_obj_MCP_LM(N, X, beta_md, y, _lambda, gamma, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X @ x_temp
    return (error.T @ error) / (2 * N) + cp.sum(
        SCAD(x_temp, lambda_=_lambda, gamma=gamma))


def lambda_max_LM(X, y):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    #     X_temp = X.copy()
    #     X_temp = X_temp[:,1:]
    #     X_temp -= cp.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= cp.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= cp.mean(y)
    #     y_temp /= cp.std(y)
    grad_at_0 = y @ X[:, 1:] / len(y)
    lambda_max = cp.linalg.norm(grad_at_0, ord=cp.infty)
    return lambda_max


def strong_rule_seq_LM(X, y, beta_old, lambda_new, lambda_old):
    """
    Use sequential strong to determine which betas to be nullified next.
    """
    #     X_temp = X.copy()
    #     X_temp -= cp.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= cp.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= cp.mean(y)
    #     y_temp /= cp.std(y)
    grad = cp.abs((y - X[:, 1:] @ beta_old[1:]) @ X[:, 1:] / (2 * len(y)))
    # True means the value gets eliminated
    eliminated = (grad < 2 * lambda_new - lambda_old)
    # because intercept coefficient is not penalized
    eliminated = cp.hstack((cp.array([False]), eliminated))
    return eliminated


def UAG_LM_SCAD_MCP(design_matrix,
                    outcome,
                    beta_0=cp.ones(1),
                    tol=1e-2,
                    maxit=500,
                    _lambda=.5,
                    penalty="SCAD",
                    a=3.7,
                    gamma=2.,
                    L_convex=1.1,
                    add_intercept_column=True):
    '''
    Carry out the optimization for penalized LM for a fixed lambda.
    '''
    X = design_matrix.copy()
    y = outcome.copy()
    N = X.shape[0]
    if cp.all(beta_0 == cp.ones(1)):
        cov = (y - cp.mean(y)) @ (X - 1 / N * cp.sum(X, 0).reshape(1, -1))
        beta = cp.sign(cov)
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        # check if design matrix has included a column for intercept or not
        if cp.any(X[:, 0] != X[0, 0]):
            intercept_design = cp.ones(N).reshape(-1, 1)
            X = cp.hstack((intercept_design, X))
            beta = cp.hstack((cp.array([0.]), beta))
    # passing other parameters
    p = X.shape[1]  # so here p includes the intercept design matrix column
    smooth_grad = cp.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    #     L_convex = 1/N*cp.max(cp.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1 / N * (cp.linalg.eigvalsh(X @ X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        L = max([L_convex, 1. / (a - 1)])
        #         L = cp.linalg.norm(cp.array([L_convex, 1./(a-1)]), ord=cp.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                opt_alpha = 2 / (1 + (1 + 4. / opt_alpha**2)**.5)
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = cp.linalg.norm(beta_md - beta_md_old,
                                        ord=2)  # restarting
            converged = (cp.linalg.norm(beta_md - beta_md_old, ord=cp.infty) <
                         tol)
            smooth_grad = update_smooth_grad_SCAD_LM(N=N,
                                                     X=X,
                                                     beta_md=beta_md,
                                                     y=y,
                                                     _lambda=_lambda,
                                                     a=a)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = cp.all(cp.max(cp.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (cp.linalg.norm(beta_md - beta_ag, ord=cp.infty) < (tol*opt_beta))
    else:
        L = max([L_convex, 1. / gamma])
        #         L = cp.linalg.norm(cp.array([L_convex, 1./(gamma)]), ord=cp.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                opt_alpha = 2 / (1 + (1 + 4. / opt_alpha**2)**.5)
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = cp.linalg.norm(beta_md - beta_md_old,
                                        ord=2)  # restarting
            converged = (cp.linalg.norm(beta_md - beta_md_old, ord=cp.infty) <
                         tol)
            smooth_grad = update_smooth_grad_MCP_LM(N=N,
                                                    X=X,
                                                    beta_md=beta_md,
                                                    y=y,
                                                    _lambda=_lambda,
                                                    gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = cp.all(cp.max(cp.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (cp.linalg.norm(beta_md - beta_ag, ord=cp.infty) < (tol*opt_beta))
    return k, beta_md


# def vanilla_proximal(self):
#     '''
#     Carry out optimization using vanilla gradient descent.
#     '''
#     if self.penalty == "SCAD":
#         L = max([self.L_convex, 1/(self.a-1)])
#         self.vanilla_stepsize = 1/L
#         self.eval_obj_SCAD_LM(self.beta_md, self.obj_value)
#         self.eval_obj_SCAD_LM(self.beta, self.obj_value_ORIGINAL)
#         self.eval_obj_SCAD_LM(self.beta_ag, self.obj_value_AG)
#         self.old_beta = self.beta_md - 10.
#         while not self.converged:
#             self.k += 1
#             if self.k <= self.maxit:
#                 self.update_smooth_grad_SCAD_LM()
#                 self.beta_md = self.soft_thresholding(self.beta_md - self.vanilla_stepsize*self.smooth_grad, self.vanilla_stepsize*self._lambda)
#                 self.converged = cp.all(cp.max(cp.abs(self.beta_md - self.old_beta)) < self.tol).item()
#                 self.old_beta = self.beta_md.copy()
#                 self.eval_obj_SCAD_LM(self.beta_md, self.obj_value)
#                 self.eval_obj_SCAD_LM(self.beta, self.obj_value_ORIGINAL)
#                 self.eval_obj_SCAD_LM(self.beta_ag, self.obj_value_AG)
#             else:
#                 break
#     else:
#         L = max([self.L_convex, 1/self.gamma])
#         self.vanilla_stepsize = 1/L
#         self.eval_obj_MCP_LM(self.beta_md, self.obj_value)
#         self.eval_obj_MCP_LM(self.beta, self.obj_value_ORIGINAL)
#         self.eval_obj_MCP_LM(self.beta_ag, self.obj_value_AG)
#         self.old_beta = self.beta_md - 10.
#         while not self.converged:
#             self.k += 1
#             if self.k <= self.maxit:
#                 self.update_smooth_grad_MCP_LM()
#                 self.beta_md = self.soft_thresholding(self.beta_md - self.vanilla_stepsize*self.smooth_grad, self.vanilla_stepsize*self._lambda)
#                 self.converged = cp.all(cp.max(cp.abs(self.beta_md - self.old_beta)) < self.tol).item()
#                 self.old_beta = self.beta_md.copy()
#                 self.eval_obj_MCP_LM(self.beta_md, self.obj_value)
#                 self.eval_obj_MCP_LM(self.beta, self.obj_value_ORIGINAL)
#                 self.eval_obj_MCP_LM(self.beta_ag, self.obj_value_AG)
#             else:
#                 break
#     return self.report_results()


def solution_path_LM(design_matrix,
                     outcome,
                     lambda_,
                     beta_0=cp.ones(1),
                     tol=1e-2,
                     maxit=500,
                     penalty="SCAD",
                     a=3.7,
                     gamma=2.,
                     add_intercept_column=True):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    #     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        # check if design matrix has included a column for intercept or not
        if cp.any(X[:, 0] != X[0, 0]):
            intercept_design = cp.ones(N).reshape(-1, 1)
            _design_matrix = design_matrix.copy()
            _design_matrix = cp.hstack((intercept_design, _design_matrix))
    beta_mat = cp.zeros((len(lambda_) + 1, _design_matrix.shape[1]))
    for j in range(len(lambda_)):
        beta_mat[j + 1, :] = UAG_LM_SCAD_MCP(design_matrix=_design_matrix,
                                             outcome=outcome,
                                             beta_0=beta_mat[j, :],
                                             tol=tol,
                                             maxit=maxit,
                                             _lambda=lambda_[j],
                                             penalty=penalty,
                                             a=a,
                                             gamma=gamma,
                                             add_intercept_column=False)[1]
    return beta_mat[1:, :]


# with strong rule
def _UAG_LM_SCAD_MCP_strongrule(design_matrix,
                                outcome,
                                beta_0=cp.ones(1),
                                tol=1e-2,
                                maxit=500,
                                _lambda=.5,
                                penalty="SCAD",
                                a=3.7,
                                gamma=2.,
                                L_convex=1.1,
                                add_intercept_column=True,
                                strongrule=True):
    '''
    Carry out the optimization for a fixed lambda with strong rule.
    '''
    X = design_matrix.copy()
    y = outcome.copy()
    N = X.shape[0]
    if cp.all(beta_0 == cp.ones(1)):
        cov = (y - cp.mean(y)) @ (X - 1 / N * cp.sum(X, 0).reshape(1, -1))
        beta = cp.sign(cov)
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        # check if design matrix has included a column for intercept or not
        if cp.any(X[:, 0] != X[0, 0]):
            intercept_design = cp.ones(N).reshape(-1, 1)
            X = cp.hstack((intercept_design, X))
            beta = cp.hstack((cp.array([0.]), beta))
    if strongrule == True:
        _lambda_max = lambda_max_LM(X, y)
        p_original = X.shape[1]
        elim = strong_rule_seq_LM(X,
                                  y,
                                  beta_old=cp.zeros(p_original),
                                  lambda_new=_lambda,
                                  lambda_old=_lambda_max)
        X = X[:, cp.logical_not(elim)]
        beta = beta[cp.logical_not(elim)]

    # passing other parameters
    p = X.shape[1]  # so here p includes the intercept design matrix column
    smooth_grad = cp.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    #     L_convex = 1/N*cp.max(cp.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1 / N * (cp.linalg.eigvalsh(X @ X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        L = max([L_convex, 1 / (a - 1)])
        #         L = cp.linalg.norm(cp.array([L_convex, 1./(a-1)]), ord=cp.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                opt_alpha = 2. / (1. + (1. + 4. / opt_alpha**2)**.5)
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1. - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = cp.linalg.norm(beta_md - beta_md_old,
                                        ord=2)  # restarting
            converged = (cp.linalg.norm(beta_md - beta_md_old, ord=cp.infty) <
                         tol)
            smooth_grad = update_smooth_grad_SCAD_LM(N=N,
                                                     X=X,
                                                     beta_md=beta_md,
                                                     y=y,
                                                     _lambda=_lambda,
                                                     a=a)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = cp.all(cp.max(cp.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (cp.linalg.norm(beta_md - beta_ag, ord=cp.infty) < (tol*opt_beta))
    else:
        L = max([L_convex, 1 / gamma])
        #         L = cp.linalg.norm(cp.array([L_convex, 1./(gamma)]), ord=cp.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                opt_alpha = 2 / (1. + (1. + 4. / opt_alpha**2)**.5)
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1. - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = cp.linalg.norm(beta_md - beta_md_old,
                                        ord=2)  # restarting
            converged = (cp.linalg.norm(beta_md - beta_md_old, ord=cp.infty) <
                         tol)
            smooth_grad = update_smooth_grad_MCP_LM(N=N,
                                                    X=X,
                                                    beta_md=beta_md,
                                                    y=y,
                                                    _lambda=_lambda,
                                                    gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)


#             converged = cp.all(cp.max(cp.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (cp.linalg.norm(beta_md - beta_ag, ord=cp.infty) < (tol*opt_beta))
#     if strongrule == True:
#         _beta_output = cp.zeros((p_original))
# #         _ = cp.argwhere(cp.logical_not(elim)).flatten()
# #         print(_)
# #         for j in range(len(_)):
# #             if j<10:
# #                 print(j)
# #                 print(_[j])
# #             _beta_output[_[j]] = beta_md[j]
# #             if j<10:
# #                 print(_beta_output[_[j]])
#         _beta_output[~elim] = beta_md  # this line of code can't compile
#     else:
#         _beta_output = beta_md
    return k, beta_md, elim


def UAG_LM_SCAD_MCP_strongrule(design_matrix,
                               outcome,
                               beta_0=cp.ones(1),
                               tol=1e-2,
                               maxit=500,
                               _lambda=.5,
                               penalty="SCAD",
                               a=3.7,
                               gamma=2.,
                               L_convex=1.1,
                               add_intercept_column=True,
                               strongrule=True):
    """
    Carry out the optimization for a fixed lambda for penanlized LM with strong rule.
    """
    _k, _beta_md, _elim = _UAG_LM_SCAD_MCP_strongrule(
        design_matrix=design_matrix,
        outcome=outcome,
        beta_0=beta_0,
        tol=tol,
        maxit=maxit,
        _lambda=_lambda,
        penalty=penalty,
        a=a,
        gamma=gamma,
        L_convex=L_convex,
        add_intercept_column=add_intercept_column,
        strongrule=strongrule)
    output_beta = cp.zeros(len(_elim))
    output_beta[cp.logical_not(_elim)] = _beta_md
    return _k, output_beta


def solution_path_LM_strongrule(design_matrix,
                                outcome,
                                lambda_,
                                beta_0=cp.ones(1),
                                tol=1e-2,
                                maxit=500,
                                penalty="SCAD",
                                a=3.7,
                                gamma=2.,
                                add_intercept_column=True):
    '''
    Carry out the optimization for the solution path of a penalized LM with strong rule.
    '''
    #     add design matrix column for the intercept, if it's not there already
    _design_matrix = design_matrix.copy()
    if add_intercept_column == True:
        # check if design matrix has included a column for intercept or not
        if cp.any(design_matrix[:, 0] != design_matrix[0, 0]):
            intercept_design = cp.ones(N).reshape(-1, 1)
            _design_matrix = cp.hstack((intercept_design, _design_matrix))
    beta_mat = cp.empty((len(lambda_) + 1, _design_matrix.shape[1]))
    beta_mat[0, :] = 0.
    _lambda_max = lambda_max_LM(_design_matrix, outcome)
    lambda_ = cp.hstack((cp.array([_lambda_max]), lambda_))
    elim = cp.array([False] * _design_matrix.shape[1])
    for j in range(len(lambda_) - 1):
        _elim = strong_rule_seq_LM(X=_design_matrix,
                                   y=outcome,
                                   beta_old=beta_mat[j, :],
                                   lambda_new=lambda_[j + 1],
                                   lambda_old=lambda_[j])
        elim = cp.logical_and(elim, _elim)
        _beta_0 = beta_mat[j, :]
        _new_beta = cp.zeros(_design_matrix.shape[1])
        _new_beta[cp.logical_not(elim)] = UAG_LM_SCAD_MCP(
            design_matrix=_design_matrix[:, cp.logical_not(elim)],
            outcome=outcome,
            beta_0=_beta_0[cp.logical_not(elim)],
            tol=tol,
            maxit=maxit,
            _lambda=lambda_[j],
            penalty=penalty,
            a=a,
            gamma=gamma,
            add_intercept_column=False)[1]
        beta_mat[j + 1, :] = _new_beta
    return beta_mat[1:, :]


def update_smooth_grad_convex_logistic(N, X, beta_md, y):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    return (X.T @ (cp.tanh(X @ beta_md / 2.) / 2. - y + .5)) / (2. * N)


def update_smooth_grad_convex_logistic(N, X, beta_md, y):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    return 1. / N * X.T @ (X @ beta_md - y)


def update_smooth_grad_SCAD_logistic(N, X, beta_md, y, _lambda, a):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return update_smooth_grad_convex_logistic(
        N=N, X=X, beta_md=beta_md, y=y) + SCAD_concave_grad(
            x=beta_md, lambda_=_lambda, a=a)


def update_smooth_grad_MCP_logistic(N, X, beta_md, y, _lambda, gamma):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return update_smooth_grad_convex_logistic(
        N=N, X=X, beta_md=beta_md, y=y) + MCP_concave_grad(
            x=beta_md, lambda_=_lambda, gamma=gamma)


def eval_obj_SCAD_logistic(N, X, beta_md, y, _lambda, a, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X @ x_temp
    return (error.T @ error) / (2. * N) + cp.sum(
        SCAD(x_temp, lambda_=_lambda, a=a))


def eval_obj_MCP_logistic(N, X, beta_md, y, _lambda, gamma, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X @ x_temp
    return (error.T @ error) / (2 * N) + cp.sum(
        SCAD(x_temp, lambda_=_lambda, gamma=gamma))


def lambda_max_logistic(X, y):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    #     X_temp = X.copy()
    #     X_temp = X_temp[:,1:]
    #     X_temp -= cp.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= cp.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= cp.mean(y)
    #     y_temp /= cp.std(y)
    grad_at_0 = y @ X[:, 1:] / len(y)
    lambda_max = cp.linalg.norm(grad_at_0, ord=cp.infty)
    return lambda_max


def strong_rule_seq_logistic(X, y, beta_old, lambda_new, lambda_old):
    """
    Use sequential strong to determine which betas to be nullified next.
    """
    #     X_temp = X.copy()
    #     X_temp -= cp.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= cp.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= cp.mean(y)
    #     y_temp /= cp.std(y)
    grad = cp.abs((y - X[:, 1:] @ beta_old[1:]) @ X[:, 1:] / (2 * len(y)))
    # True means the value gets eliminated
    eliminated = (grad < 2 * lambda_new - lambda_old)
    # because intercept coefficient is not penalized
    eliminated = cp.hstack((cp.array([False]), eliminated))
    return eliminated


def UAG_logistic_SCAD_MCP(design_matrix,
                          outcome,
                          beta_0=cp.ones(1),
                          tol=1e-2,
                          maxit=500,
                          _lambda=.5,
                          penalty="SCAD",
                          a=3.7,
                          gamma=2.,
                          L_convex=1.1,
                          add_intercept_column=True):
    '''
    Carry out the optimization for penalized logistic model for a fixed lambda.
    '''
    X = design_matrix.copy()
    y = outcome.copy()
    N = X.shape[0]
    if cp.all(beta_0 == cp.ones(1)):
        cov = (y - cp.mean(y)) @ (X - 1 / N * cp.sum(X, 0).reshape(1, -1))
        beta = cp.sign(cov)
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        # check if design matrix has included a column for intercept or not
        if cp.any(X[:, 0] != X[0, 0]):
            intercept_design = cp.ones(N).reshape(-1, 1)
            X = cp.hstack((intercept_design, X))
            beta = cp.hstack((cp.array([0.]), beta))
    # passing other parameters
    p = X.shape[1]  # so here p includes the intercept design matrix column
    smooth_grad = cp.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    #     L_convex = 1/N*cp.max(cp.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1 / N * (cp.linalg.eigvalsh(X @ X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        L = max([L_convex, 1. / (a - 1)])
        #         L = cp.linalg.norm(cp.array([L_convex, 1./(a-1)]), ord=cp.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                opt_alpha = 2 / (1 + (1 + 4. / opt_alpha**2)**.5)
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = cp.linalg.norm(beta_md - beta_md_old,
                                        ord=2)  # restarting
            converged = (cp.linalg.norm(beta_md - beta_md_old, ord=cp.infty) <
                         tol)
            smooth_grad = update_smooth_grad_SCAD_logistic(N=N,
                                                           X=X,
                                                           beta_md=beta_md,
                                                           y=y,
                                                           _lambda=_lambda,
                                                           a=a)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = cp.all(cp.max(cp.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (cp.linalg.norm(beta_md - beta_ag, ord=cp.infty) < (tol*opt_beta))
    else:
        L = max([L_convex, 1. / gamma])
        #         L = cp.linalg.norm(cp.array([L_convex, 1./(gamma)]), ord=cp.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                opt_alpha = 2 / (1 + (1 + 4. / opt_alpha**2)**.5)
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = cp.linalg.norm(beta_md - beta_md_old,
                                        ord=2)  # restarting
            converged = (cp.linalg.norm(beta_md - beta_md_old, ord=cp.infty) <
                         tol)
            smooth_grad = update_smooth_grad_MCP_logistic(N=N,
                                                          X=X,
                                                          beta_md=beta_md,
                                                          y=y,
                                                          _lambda=_lambda,
                                                          gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = cp.all(cp.max(cp.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (cp.linalg.norm(beta_md - beta_ag, ord=cp.infty) < (tol*opt_beta))
    return k, beta_md


# def vanilla_proximal(self):
#     '''
#     Carry out optimization using vanilla gradient descent.
#     '''
#     if self.penalty == "SCAD":
#         L = max([self.L_convex, 1/(self.a-1)])
#         self.vanilla_stepsize = 1/L
#         self.eval_obj_SCAD_logistic(self.beta_md, self.obj_value)
#         self.eval_obj_SCAD_logistic(self.beta, self.obj_value_ORIGINAL)
#         self.eval_obj_SCAD_logistic(self.beta_ag, self.obj_value_AG)
#         self.old_beta = self.beta_md - 10.
#         while not self.converged:
#             self.k += 1
#             if self.k <= self.maxit:
#                 self.update_smooth_grad_SCAD_logistic()
#                 self.beta_md = self.soft_thresholding(self.beta_md - self.vanilla_stepsize*self.smooth_grad, self.vanilla_stepsize*self._lambda)
#                 self.converged = cp.all(cp.max(cp.abs(self.beta_md - self.old_beta)) < self.tol).item()
#                 self.old_beta = self.beta_md.copy()
#                 self.eval_obj_SCAD_logistic(self.beta_md, self.obj_value)
#                 self.eval_obj_SCAD_logistic(self.beta, self.obj_value_ORIGINAL)
#                 self.eval_obj_SCAD_logistic(self.beta_ag, self.obj_value_AG)
#             else:
#                 break
#     else:
#         L = max([self.L_convex, 1/self.gamma])
#         self.vanilla_stepsize = 1/L
#         self.eval_obj_MCP_logistic(self.beta_md, self.obj_value)
#         self.eval_obj_MCP_logistic(self.beta, self.obj_value_ORIGINAL)
#         self.eval_obj_MCP_logistic(self.beta_ag, self.obj_value_AG)
#         self.old_beta = self.beta_md - 10.
#         while not self.converged:
#             self.k += 1
#             if self.k <= self.maxit:
#                 self.update_smooth_grad_MCP_logistic()
#                 self.beta_md = self.soft_thresholding(self.beta_md - self.vanilla_stepsize*self.smooth_grad, self.vanilla_stepsize*self._lambda)
#                 self.converged = cp.all(cp.max(cp.abs(self.beta_md - self.old_beta)) < self.tol).item()
#                 self.old_beta = self.beta_md.copy()
#                 self.eval_obj_MCP_logistic(self.beta_md, self.obj_value)
#                 self.eval_obj_MCP_logistic(self.beta, self.obj_value_ORIGINAL)
#                 self.eval_obj_MCP_logistic(self.beta_ag, self.obj_value_AG)
#             else:
#                 break
#     return self.report_results()


def solution_path_logistic(design_matrix,
                           outcome,
                           lambda_,
                           beta_0=cp.ones(1),
                           tol=1e-2,
                           maxit=500,
                           penalty="SCAD",
                           a=3.7,
                           gamma=2.,
                           add_intercept_column=True):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    #     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        # check if design matrix has included a column for intercept or not
        if cp.any(X[:, 0] != X[0, 0]):
            intercept_design = cp.ones(N).reshape(-1, 1)
            _design_matrix = design_matrix.copy()
            _design_matrix = cp.hstack((intercept_design, _design_matrix))
    beta_mat = cp.zeros((len(lambda_) + 1, _design_matrix.shape[1]))
    for j in range(len(lambda_)):
        beta_mat[j + 1, :] = UAG_logistic_SCAD_MCP(
            design_matrix=_design_matrix,
            outcome=outcome,
            beta_0=beta_mat[j, :],
            tol=tol,
            maxit=maxit,
            _lambda=lambda_[j],
            penalty=penalty,
            a=a,
            gamma=gamma,
            add_intercept_column=False)[1]
    return beta_mat[1:, :]


# with strong rule
def _UAG_logistic_SCAD_MCP_strongrule(design_matrix,
                                      outcome,
                                      beta_0=cp.ones(1),
                                      tol=1e-2,
                                      maxit=500,
                                      _lambda=.5,
                                      penalty="SCAD",
                                      a=3.7,
                                      gamma=2.,
                                      L_convex=1.1,
                                      add_intercept_column=True,
                                      strongrule=True):
    '''
    Carry out the optimization for a fixed lambda with strong rule.
    '''
    X = design_matrix.copy()
    y = outcome.copy()
    N = X.shape[0]
    if cp.all(beta_0 == cp.ones(1)):
        cov = (y - cp.mean(y)) @ (X - 1 / N * cp.sum(X, 0).reshape(1, -1))
        beta = cp.sign(cov)
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        # check if design matrix has included a column for intercept or not
        if cp.any(X[:, 0] != X[0, 0]):
            intercept_design = cp.ones(N).reshape(-1, 1)
            X = cp.hstack((intercept_design, X))
            beta = cp.hstack((cp.array([0.]), beta))
    if strongrule == True:
        _lambda_max = lambda_max_logistic(X, y)
        p_original = X.shape[1]
        elim = strong_rule_seq_logistic(X,
                                        y,
                                        beta_old=cp.zeros(p_original),
                                        lambda_new=_lambda,
                                        lambda_old=_lambda_max)
        X = X[:, cp.logical_not(elim)]
        beta = beta[cp.logical_not(elim)]

    # passing other parameters
    p = X.shape[1]  # so here p includes the intercept design matrix column
    smooth_grad = cp.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    #     L_convex = 1/N*cp.max(cp.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1 / N * (cp.linalg.eigvalsh(X @ X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        L = max([L_convex, 1 / (a - 1)])
        #         L = cp.linalg.norm(cp.array([L_convex, 1./(a-1)]), ord=cp.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                opt_alpha = 2. / (1. + (1. + 4. / opt_alpha**2)**.5)
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1. - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = cp.linalg.norm(beta_md - beta_md_old,
                                        ord=2)  # restarting
            converged = (cp.linalg.norm(beta_md - beta_md_old, ord=cp.infty) <
                         tol)
            smooth_grad = update_smooth_grad_SCAD_logistic(N=N,
                                                           X=X,
                                                           beta_md=beta_md,
                                                           y=y,
                                                           _lambda=_lambda,
                                                           a=a)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = cp.all(cp.max(cp.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (cp.linalg.norm(beta_md - beta_ag, ord=cp.infty) < (tol*opt_beta))
    else:
        L = max([L_convex, 1 / gamma])
        #         L = cp.linalg.norm(cp.array([L_convex, 1./(gamma)]), ord=cp.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
                opt_alpha = 2 / (1. + (1. + 4. / opt_alpha**2)**.5)
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1. - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = cp.linalg.norm(beta_md - beta_md_old,
                                        ord=2)  # restarting
            converged = (cp.linalg.norm(beta_md - beta_md_old, ord=cp.infty) <
                         tol)
            smooth_grad = update_smooth_grad_MCP_logistic(N=N,
                                                          X=X,
                                                          beta_md=beta_md,
                                                          y=y,
                                                          _lambda=_lambda,
                                                          gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)


#             converged = cp.all(cp.max(cp.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (cp.linalg.norm(beta_md - beta_ag, ord=cp.infty) < (tol*opt_beta))
#     if strongrule == True:
#         _beta_output = cp.zeros((p_original))
# #         _ = cp.argwhere(cp.logical_not(elim)).flatten()
# #         print(_)
# #         for j in range(len(_)):
# #             if j<10:
# #                 print(j)
# #                 print(_[j])
# #             _beta_output[_[j]] = beta_md[j]
# #             if j<10:
# #                 print(_beta_output[_[j]])
#         _beta_output[~elim] = beta_md  # this line of code can't compile
#     else:
#         _beta_output = beta_md
    return k, beta_md, elim


def UAG_logistic_SCAD_MCP_strongrule(design_matrix,
                                     outcome,
                                     beta_0=cp.ones(1),
                                     tol=1e-2,
                                     maxit=500,
                                     _lambda=.5,
                                     penalty="SCAD",
                                     a=3.7,
                                     gamma=2.,
                                     L_convex=1.1,
                                     add_intercept_column=True,
                                     strongrule=True):
    """
    Carry out the optimization for a fixed lambda for penanlized logistic model with strong rule.
    """
    _k, _beta_md, _elim = _UAG_logistic_SCAD_MCP_strongrule(
        design_matrix=design_matrix,
        outcome=outcome,
        beta_0=beta_0,
        tol=tol,
        maxit=maxit,
        _lambda=_lambda,
        penalty=penalty,
        a=a,
        gamma=gamma,
        L_convex=L_convex,
        add_intercept_column=add_intercept_column,
        strongrule=strongrule)
    output_beta = cp.zeros(len(_elim))
    output_beta[cp.logical_not(_elim)] = _beta_md
    return _k, output_beta


def solution_path_logistic_strongrule(design_matrix,
                                      outcome,
                                      lambda_,
                                      beta_0=cp.ones(1),
                                      tol=1e-2,
                                      maxit=500,
                                      penalty="SCAD",
                                      a=3.7,
                                      gamma=2.,
                                      add_intercept_column=True):
    '''
    Carry out the optimization for the solution path of a penalized logistic model with strong rule.
    '''
    #     add design matrix column for the intercept, if it's not there already
    _design_matrix = design_matrix.copy()
    if add_intercept_column == True:
        # check if design matrix has included a column for intercept or not
        if cp.any(design_matrix[:, 0] != design_matrix[0, 0]):
            intercept_design = cp.ones(N).reshape(-1, 1)
            _design_matrix = cp.hstack((intercept_design, _design_matrix))
    beta_mat = cp.empty((len(lambda_) + 1, _design_matrix.shape[1]))
    beta_mat[0, :] = 0.
    _lambda_max = lambda_max_logistic(_design_matrix, outcome)
    lambda_ = cp.hstack((cp.array([_lambda_max]), lambda_))
    elim = cp.array([False] * _design_matrix.shape[1])
    for j in range(len(lambda_) - 1):
        _elim = strong_rule_seq_logistic(X=_design_matrix,
                                         y=outcome,
                                         beta_old=beta_mat[j, :],
                                         lambda_new=lambda_[j + 1],
                                         lambda_old=lambda_[j])
        elim = cp.logical_and(elim, _elim)
        _beta_0 = beta_mat[j, :]
        _new_beta = cp.zeros(_design_matrix.shape[1])
        _new_beta[cp.logical_not(elim)] = UAG_logistic_SCAD_MCP(
            design_matrix=_design_matrix[:, cp.logical_not(elim)],
            outcome=outcome,
            beta_0=_beta_0[cp.logical_not(elim)],
            tol=tol,
            maxit=maxit,
            _lambda=lambda_[j],
            penalty=penalty,
            a=a,
            gamma=gamma,
            add_intercept_column=False)[1]
        beta_mat[j + 1, :] = _new_beta
    return beta_mat[1:, :]
