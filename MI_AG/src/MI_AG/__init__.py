#!/usr/bin/env python
# coding: utf-8

import numpy as np
from bed_reader import open_bed
from KDEpy import FFTKDE
import multiprocess as mp
from numba import jit, njit


#############################################################################
################# filtering using mutual information ########################
#############################################################################

def MI_continuous(a, b, a_min, a_max, N=500):
    """
    calculate mutual information between continuous outcome and an SNP variable of 0,1,2
    assume no missing data
    """
    # first estimate the pmf
    p0 = np.sum(b==0)/len(b)
    p1 = np.sum(b==1)/len(b)
    p2 = 1.-p0-p1
    # estimate cond density
    _b0 = (b==0)
    if np.sum(_b0)>2: # here proceed to kde only if there are more than 5 data points
        y_cond_p0 = FFTKDE(kernel="gaussian", bw="silverman").fit(data=a[_b0])
#         y_cond_p0 = gaussian_kde(a[_b0])
    else:
        y_cond_p0 = np.zeros_like
    _b1 = (b==1)
    if np.sum(_b1)>2:
        y_cond_p1 = FFTKDE(kernel="gaussian", bw="silverman").fit(data=a[_b1])
#         y_cond_p1 = gaussian_kde(a[_b1]) # this thing uses Scott's rule instead of Silverman defaulted by FFTKDE and R density
    else:
        y_cond_p1 = np.zeros_like
    _b2 = (b==2)
    if np.sum(_b2)>2:
        y_cond_p2 = FFTKDE(kernel="gaussian", bw="silverman").fit(data=a[_b2])
#         y_cond_p2 = gaussian_kde(a[_b2])
    else:
        y_cond_p2 = np.zeros_like
    joint = np.empty((N,3))
    a_temp = np.linspace(a_min, a_max, num=N)
    joint[:,0] = y_cond_p0(a_temp)*p0
    joint[:,1] = y_cond_p1(a_temp)*p1
    joint[:,2] = y_cond_p2(a_temp)*p2
    joint[joint<1e-20] = 1e-20 # set a threshold to avoid numerical errors
    forward_euler_step = a_temp[1]-a_temp[0]
#     print("total measure:", np.sum(joint)*forward_euler_step)
    temp_log = np.log(joint)
#     temp_log = np.nan_to_num(temp_log, nan = 0)
    temp1 = np.log(np.sum(joint, 1))
#     temp1 = np.nan_to_num(temp1, nan = 0)
    temp_log = temp_log - temp1.reshape(-1,1)
    temp2 = np.log(np.sum(joint, 0)*forward_euler_step)
#     temp2 = np.nan_to_num(temp2, nan = 0)
    temp_log = temp_log - temp2.reshape(1,-1)
    # print(fhat_mat * temp_log)
    temp_mat = joint * temp_log
#     temp_mat = np.nan_to_num(temp_mat, nan=0.) # numerical fix
    mi_temp = np.sum(temp_mat)*forward_euler_step
    return mi_temp

def MI_binary(a, b, a_min, a_max, N=500):
    """
    calculate mutual information between binary outcome and an SNP variable of 0,1,2
    assume no missing data
    """
    # first estimate the pmf of SNP
    p0 = np.sum(b==0)/len(b)
    p1 = np.sum(b==1)/len(b)
    p2 = 1.-p0-p1
    # estimate pmf of the binary outcome
    a_p0 = np.sum(a==0)/len(a)
    a_p1 = np.sum(a==1)/len(a)
    # estimate cond density
    _b0 = (b==0)
    if np.sum(_b0)>2: # here proceed to kde only if there are more than 5 data points
        y_cond_p0 = lambda x: x*np.sum(a[_b0]==1)/len(a[_b0]) + (1-x)*np.sum(a[_b0]==0)/len(a[_b0])
#         y_cond_p0 = gaussian_kde(a[_b0])
    else:
        y_cond_p0 = np.zeros_like
    _b1 = (b==1)
    if np.sum(_b1)>2:
        y_cond_p1 = lambda x: x*np.sum(a[_b1]==1)/len(a[_b1]) + (1-x)*np.sum(a[_b1]==0)/len(a[_b1])
#         y_cond_p1 = gaussian_kde(a[_b1]) # this thing uses Scott's rule instead of Silverman defaulted by FFTKDE and R density
    else:
        y_cond_p1 = np.zeros_like
    _b2 = (b==2)
    if np.sum(_b2)>2:
        y_cond_p2 = lambda x: x*np.sum(a[_b2]==1)/len(a[_b2]) + (1-x)*np.sum(a[_b2]==0)/len(a[_b2])
#         y_cond_p2 = gaussian_kde(a[_b2])
    else:
        y_cond_p2 = np.zeros_like
    joint = np.empty((N,3))
    a_temp = np.linspace(a_min, a_max, num=N)
    joint[:,0] = y_cond_p0(a_temp)*p0
    joint[:,1] = y_cond_p1(a_temp)*p1
    joint[:,2] = y_cond_p2(a_temp)*p2
    joint[joint<1e-20] = 1e-20 # set a threshold to avoid numerical errors
    forward_euler_step = a_temp[1]-a_temp[0]
#     print("total measure:", np.sum(joint)*forward_euler_step)
    temp_log = np.log(joint)
#     temp_log = np.nan_to_num(temp_log, nan = 0)
    temp1 = np.log(np.sum(joint, 1))
#     temp1 = np.nan_to_num(temp1, nan = 0)
    temp_log = temp_log - temp1.reshape(-1,1)
    temp2 = np.log(np.sum(joint, 0)*forward_euler_step)
#     temp2 = np.nan_to_num(temp2, nan = 0)
    temp_log = temp_log - temp2.reshape(1,-1)
    # print(fhat_mat * temp_log)
    temp_mat = joint * temp_log
#     temp_mat = np.nan_to_num(temp_mat, nan=0.) # numerical fix
    mi_temp = np.sum(temp_mat)*forward_euler_step
    return mi_temp

# outcome_iid should be a  list of strings for identifiers 
def continuous_filter(bed_file, bim_file, fam_file, outcome, outcome_iid, a_min=100., a_max=250., N=500):
    """
    (Single Core version) take plink files to calculate the mutual information between the continuous outcome and many SNP variables.
    """
    bed1 = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    gene_iid = np.array(list(bed1.iid))
    bed1_sid = np.array(list(bed1.sid))
    outcome = outcome[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
    # get genetic indices
    gene_ind = np.intersect1d(gene_iid, outcome_iid, assume_unique=True, return_indices=True)[1]
    MI_UKBB = np.zeros_like(bed1_sid)
    for j in range(len(MI_UKBB)):
        _SNP = bed1.read(np.s_[:,j], dtype=np.int8).flatten()
        _SNP = _SNP[gene_ind] # get gene iid also in outcome iid
        _outcome = outcome[_SNP != -127] # remove missing SNP in outcome
        _SNP = _SNP[_SNP != -127] # remove missing SNP
        MI_UKBB[j] = MI_continuous(a=_outcome, b=_SNP, a_min = a_min, a_max = a_max, N=N)
    return MI_UKBB

def binary_filter(bed_file, bim_file, fam_file, outcome, outcome_iid, a_min=100., a_max=250., N=500):
    """
    (Single Core version) take plink files to calculate the mutual information between the binary outcome and many SNP variables.
    """
    bed1 = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    gene_iid = np.array(list(bed1.iid))
    bed1_sid = np.array(list(bed1.sid))
    outcome = outcome[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
    # get genetic indices
    gene_ind = np.intersect1d(gene_iid, outcome_iid, assume_unique=True, return_indices=True)[1]
    MI_UKBB = np.zeros_like(bed1_sid)
    for j in range(len(MI_UKBB)):
        _SNP = bed1.read(np.s_[:,j], dtype=np.int8).flatten()
        _SNP = _SNP[gene_ind] # get gene iid also in outcome iid
        _outcome = outcome[_SNP != -127] # remove missing SNP in outcome
        _SNP = _SNP[_SNP != -127] # remove missing SNP
        MI_UKBB[j] = MI_binary(a=_outcome, b=_SNP, a_min = a_min, a_max = a_max, N=N)
    return MI_UKBB


def continuous_filter_parallel(bed_file, bim_file, fam_file, outcome, outcome_iid, a_min=100., a_max=250., N=500, chunck_size=60000):
    """
    (Multiprocessing version) take plink files to calculate the mutual information between the continuous outcome and many SNP variables.
    """
    bed1 = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    gene_iid = np.array(list(bed1.iid))
    bed1_sid = np.array(list(bed1.sid))
    outcome = outcome[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
    # get genetic indices
    gene_ind = np.intersect1d(gene_iid, outcome_iid, assume_unique=True, return_indices=True)[1]
    def _continuous_filter_slice(_slice):
        _MI_slice = np.zeros_like(_slice)
        k = 0
        for j in _slice:
            _SNP = bed1.read(np.s_[:,j], dtype=np.int8).flatten()
            _SNP = _SNP[gene_ind] # get gene iid also in outcome iid
            _outcome = outcome[_SNP != -127] # remove missing SNP in outcome
            _SNP = _SNP[_SNP != -127] # remove missing SNP
            _MI_slice[k] = MI_continuous(a=_outcome, b=_SNP, a_min = a_min, a_max = a_max, N=N)
            k += 1
        return _MI_slice
    # multiprocessing starts here
    ind = np.arange(len(bed1_sid))
    n_slices = np.ceil(len(ind)/chunck_size)
    with mp.Pool(mp.cpu_count()) as p:
        MI_UKBB = p.map(_continuous_filter_slice, np.array_split(ind, n_slices))
    MI_UKBB = np.hstack(MI_UKBB)
    return MI_UKBB


def binary_filter_parallel(bed_file, bim_file, fam_file, outcome, outcome_iid, a_min=100., a_max=250., N=500, chunck_size=60000):
    """
    (Multiprocessing version) take plink files to calculate the mutual information between the binary outcome and many SNP variables.
    """
    bed1 = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    gene_iid = np.array(list(bed1.iid))
    bed1_sid = np.array(list(bed1.sid))
    outcome = outcome[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
    # get genetic indices
    gene_ind = np.intersect1d(gene_iid, outcome_iid, assume_unique=True, return_indices=True)[1]
    def _binary_filter_slice(_slice):
        _MI_slice = np.zeros_like(_slice)
        k = 0
        for j in _slice:
            _SNP = bed1.read(np.s_[:,j], dtype=np.int8).flatten()
            _SNP = _SNP[gene_ind] # get gene iid also in outcome iid
            _outcome = outcome[_SNP != -127] # remove missing SNP in outcome
            _SNP = _SNP[_SNP != -127] # remove missing SNP
            _MI_slice[k] = MI_binary(a=_outcome, b=_SNP, a_min = a_min, a_max = a_max, N=N)
            k += 1
        return _MI_slice
    # multiprocessing starts here
    ind = np.arange(len(bed1_sid))
    n_slices = np.ceil(len(ind)/chunck_size)
    with mp.Pool(mp.cpu_count()) as p:
        MI_UKBB = p.map(_binary_filter_slice, np.array_split(ind, n_slices))
    MI_UKBB = np.hstack(MI_UKBB)
    return MI_UKBB

##################################################################
################ some fudamentals things #########################
##################################################################

######################################  some SCAD and MCP things  #######################################
@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def soft_thresholding(x, lambda_):
    '''
    To calculate soft-thresholding mapping of a given ONE-DIMENSIONAL tensor, BESIDES THE FIRST TERM (so beta_0 will not be penalized). 
    This function is to be used for calculation involving L1 penalty term later. 
    '''
    return np.hstack((np.array([x[0]]), np.where(np.abs(x[1:])>lambda_, x[1:] - np.sign(x[1:])*lambda_, 0)))

soft_thresholding(np.random.rand(20),3.1)

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SCAD(x, lambda_, a=3.7):
    '''
    To calculate SCAD penalty value;
    #x can be a multi-dimensional tensor;
    lambda_, a are scalars;
    Fan and Li suggests to take a as 3.7 
    '''
    # here I notice the function is de facto a function of absolute value of x, therefore take absolute value first to simplify calculation 
    x = np.abs(x)
    temp = np.where(x<=lambda_, lambda_*x, np.where(x<a*lambda_, (2*a*lambda_*x - x**2 - lambda_**2)/(2*(a - 1)), lambda_**2 * (a+1)/2))
    temp[0] = 0. # this is to NOT penalize intercept beta later 
    return temp 

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SCAD_grad(x, lambda_, a=3.7):
    '''
    To calculate the gradient of SCAD wrt. input x; 
    #x can be a multi-dimensional tensor. 
    '''
    # here decompose x to sign and its absolute value for easier calculation
    sgn = np.sign(x)
    x = np.abs(x)
    temp = np.where(x<=lambda_, lambda_*sgn, np.where(x<a*lambda_, (a*lambda_*sgn-sgn*x)/(a-1), 0))
    temp[0] = 0. # this is to NOT penalize intercept beta later 
    return temp 

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def MCP(x, lambda_, gamma):
    '''
    To calculate MCP penalty value; 
    #x can be a multi-dimensional tensor. 
    '''
    # the function is a function of absolute value of x 
    x = np.abs(x)
    temp = np.where(x<=gamma*lambda_, lambda_*x - x**2/(2*gamma), .5*gamma*lambda_**2)
    temp[0] = 0. # this is to NOT penalize intercept beta later 
    return temp 

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def MCP_grad(x, lambda_, gamma):
    '''
    To calculate MCP gradient wrt. input x; 
    #x can be a multi-dimensional tensor. 
    '''
    temp = np.where(np.abs(x)<gamma*lambda_, lambda_*np.sign(x)-x/gamma, np.zeros_like(x))
    temp[0] = 0. # this is to NOT penalize intercept beta later 
    return temp 

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SCAD_concave(x, lambda_, a=3.7):
    '''
    The value of concave part of SCAD penalty; 
    #x can be a multi-dimensional tensor. 
    '''
    x = np.abs(x)
    temp = np.where(x<=lambda_, 0., np.where(x<a*lambda_, (lambda_*x - (x**2 + lambda_**2)/2)/(a-1), (a+1)/2*lambda_**2 - lambda_*x))
    temp[0] = 0. # this is to NOT penalize intercept beta later 
    return temp 

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SCAD_concave_grad(x, lambda_, a=3.7):
    '''
    The gradient of concave part of SCAD penalty wrt. input x; 
    #x can be a multi-dimensional tensor. 
    '''
    sgn = np.sign(x)
    x = np.abs(x)
    temp = np.where(x<=lambda_, 0., np.where(x<a*lambda_, (lambda_*sgn-sgn*x)/(a-1), -lambda_*sgn))
    temp[0] = 0. # this is to NOT penalize intercept beta later 
    return temp 

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def MCP_concave(x, lambda_, gamma):
    '''
    The value of concave part of MCP penalty; 
    #x can be a multi-dimensional tensor. 
    '''
    # similiar as in MCP
    x = np.abs(x)
    temp = np.where(x<=gamma*lambda_, -(x**2)/(2*gamma), (gamma*lambda_**2)/2 - lambda_*x)
    temp[0] = 0. # this is to NOT penalize intercept beta later 
    return temp 

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def MCP_concave_grad(x, lambda_, gamma):
    '''
    The gradient of concave part of MCP penalty wrt. input x; 
    #x can be a multi-dimensional tensor. 
    '''
    temp = np.where(np.abs(x) < gamma*lambda_, -x/gamma, -lambda_*np.sign(x))
    temp[0] = 0. # this is to NOT penalize intercept beta later 
    return temp 


##################################################################
######### LM AG normal memory version with numba #################
##################################################################

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def update_smooth_grad_convex_LM(N, X, beta_md, y):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    return 1/N*X.T@(X@beta_md - y)

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def update_smooth_grad_SCAD_LM(N, X, beta_md, y, _lambda, a):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return update_smooth_grad_convex_LM(N=N, X=X, beta_md=beta_md, y=y) + SCAD_concave_grad(x=beta_md, lambda_=_lambda, a=a)

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def update_smooth_grad_MCP_LM(N, X, beta_md, y, _lambda, gamma):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return update_smooth_grad_convex_LM(N=N, X=X, beta_md=beta_md, y=y) + MCP_concave_grad(x=beta_md, lambda_=_lambda, gamma=gamma)

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def eval_obj_SCAD_LM(N, X, beta_md, y, _lambda, a, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X@x_temp
    return (error.T@error)/(2.*N) + np.sum(SCAD(x_temp, lambda_=_lambda, a=a))

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def eval_obj_MCP_LM(N, X, beta_md, y, _lambda, gamma, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X@x_temp
    return (error.T@error)/(2*N) + np.sum(SCAD(x_temp, lambda_=_lambda, gamma=gamma))


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def lambda_max_LM(X, y):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
#     X_temp = X.copy()
#     X_temp = X_temp[:,1:]
#     X_temp -= np.mean(X_temp,0).reshape(1,-1)
#     X_temp /= np.std(X_temp,0)
#     y_temp = y.copy()
#     y_temp -= np.mean(y)
#     y_temp /= np.std(y)
    grad_at_0 = y@X[:,1:]/len(y)
    lambda_max = np.linalg.norm(grad_at_0, ord=np.infty)
    return lambda_max

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def strong_rule_seq_LM(X, y, beta_old, lambda_new, lambda_old):
    """
    Use sequential strong to determine which betas to be nullified next.
    """
#     X_temp = X.copy()
#     X_temp -= np.mean(X_temp,0).reshape(1,-1)
#     X_temp /= np.std(X_temp,0)
#     y_temp = y.copy()
#     y_temp -= np.mean(y)
#     y_temp /= np.std(y)
    grad = np.abs((y-X[:,1:]@beta_old[1:])@X[:,1:]/(2*len(y)))
    eliminated = (grad < 2*lambda_new - lambda_old) # True means the value gets eliminated
    eliminated = np.hstack((np.array([False]), eliminated)) # because intercept coefficient is not penalized
    return eliminated

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def UAG_LM_SCAD_MCP(design_matrix, outcome, beta_0 = np.ones(1), tol=1e-2, maxit=500, _lambda=.5, penalty="SCAD", a=3.7, gamma=2., L_convex=1.1, add_intercept_column = True):
    '''
    Carry out the optimization for penalized LM for a fixed lambda.
    '''
    X = design_matrix.copy()
    y = outcome.copy()
    N = X.shape[0]
    if np.all(beta_0==np.ones(1)):
        cov = (y - np.mean(y))@(X - 1/N*np.sum(X, 0).reshape(1,-1))
        beta = np.sign(cov)
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        if np.any(X[:,0] != X[0,0]): # check if design matrix has included a column for intercept or not
            intercept_design = np.ones(N).reshape(-1, 1)
            X = np.hstack((intercept_design, X))
            beta = np.hstack((np.array([0.]), beta))
    # passing other parameters
    p = X.shape[1] # so here p includes the intercept design matrix column 
    smooth_grad = np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
#     L_convex = 1/N*np.max(np.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1/N*(np.linalg.eigvalsh(X@X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0
    
    if penalty == "SCAD":
#         L = np.max(np.array([L_convex, 1./(a-1)]))
        L = np.linalg.norm(np.array([L_convex, 1./(a-1)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = update_smooth_grad_SCAD_LM(N=N, X=X, beta_md=beta_md, y=y, _lambda=_lambda, a=a)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    else:
#         L = np.max(np.array([L_convex, 1./(gamma)]))
        L = np.linalg.norm(np.array([L_convex, 1./(gamma)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = update_smooth_grad_MCP_LM(N=N, X=X, beta_md=beta_md, y=y, _lambda=_lambda, gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
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
#                 self.converged = np.all(np.max(np.abs(self.beta_md - self.old_beta)) < self.tol).item()
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
#                 self.converged = np.all(np.max(np.abs(self.beta_md - self.old_beta)) < self.tol).item()
#                 self.old_beta = self.beta_md.copy()
#                 self.eval_obj_MCP_LM(self.beta_md, self.obj_value)
#                 self.eval_obj_MCP_LM(self.beta, self.obj_value_ORIGINAL)
#                 self.eval_obj_MCP_LM(self.beta_ag, self.obj_value_AG)
#             else:
#                 break
#     return self.report_results()


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def solution_path_LM(design_matrix, outcome, lambda_, beta_0 = np.ones(1), tol=1e-2, maxit=500, penalty="SCAD", a=3.7, gamma=2., add_intercept_column=True):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    #     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        if np.any(X[:,0] != X[0,0]): # check if design matrix has included a column for intercept or not
            intercept_design = np.ones(N).reshape(-1, 1)
            _design_matrix = design_matrix.copy()
            _design_matrix = np.hstack((intercept_design, _design_matrix))
    beta_mat = np.zeros((len(lambda_)+1, _design_matrix.shape[1]))
    for j in range(len(lambda_)):
        beta_mat[j+1,:] = UAG_LM_SCAD_MCP(design_matrix=_design_matrix, outcome=outcome, beta_0 = beta_mat[j,:], tol=tol, maxit=maxit, _lambda=lambda_[j], penalty=penalty, a=a, gamma=gamma, add_intercept_column=False)[1]
    return beta_mat[1:,:]



# with strong rule 

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _UAG_LM_SCAD_MCP_strongrule(design_matrix, outcome, beta_0 = np.ones(1), tol=1e-2, maxit=500, _lambda=.5, penalty="SCAD", a=3.7, gamma=2., L_convex=1.1, add_intercept_column = True, strongrule=True):
    '''
    Carry out the optimization for a fixed lambda with strong rule.
    '''
    X = design_matrix.copy()
    y = outcome.copy()
    N = X.shape[0]
    if np.all(beta_0==np.ones(1)):
        cov = (y - np.mean(y))@(X - 1/N*np.sum(X, 0).reshape(1,-1))
        beta = np.sign(cov)
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        if np.any(X[:,0] != X[0,0]): # check if design matrix has included a column for intercept or not
            intercept_design = np.ones(N).reshape(-1, 1)
            X = np.hstack((intercept_design, X))
            beta = np.hstack((np.array([0.]), beta))
    if strongrule == True:
        _lambda_max = lambda_max_LM(X, y)
        p_original = X.shape[1]
        elim = strong_rule_seq_LM(X, y, beta_old=np.zeros(p_original), lambda_new=_lambda, lambda_old=_lambda_max)
        X = X[:, np.logical_not(elim)]
        beta = beta[np.logical_not(elim)]
        
    # passing other parameters
    p = X.shape[1] # so here p includes the intercept design matrix column 
    smooth_grad = np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
#     L_convex = 1/N*np.max(np.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1/N*(np.linalg.eigvalsh(X@X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0
    
    if penalty == "SCAD":
#         L = np.max(np.array([L_convex, 1./(a-1)]))
        L = np.linalg.norm(np.array([L_convex, 1./(a-1)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2./(1.+(1.+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1.-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = update_smooth_grad_SCAD_LM(N=N, X=X, beta_md=beta_md, y=y, _lambda=_lambda, a=a)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    else:
#         L = np.max(np.array([L_convex, 1./(gamma)]))
        L = np.linalg.norm(np.array([L_convex, 1./(gamma)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1.+(1.+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1.-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = update_smooth_grad_MCP_LM(N=N, X=X, beta_md=beta_md, y=y, _lambda=_lambda, gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
#     if strongrule == True:
#         _beta_output = np.zeros((p_original))
# #         _ = np.argwhere(np.logical_not(elim)).flatten()
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

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def UAG_LM_SCAD_MCP_strongrule(design_matrix, outcome, beta_0 = np.ones(1), tol=1e-2, maxit=500, _lambda=.5, penalty="SCAD", a=3.7, gamma=2., L_convex=1.1, add_intercept_column = True, strongrule=True):
    """
    Carry out the optimization for a fixed lambda for penanlized LM with strong rule.
    """
    _k, _beta_md, _elim = _UAG_LM_SCAD_MCP_strongrule(design_matrix=design_matrix, outcome=outcome, beta_0 = beta_0, tol=tol, maxit=maxit, _lambda=_lambda, penalty=penalty, a=a, gamma=gamma, L_convex=L_convex, add_intercept_column = add_intercept_column, strongrule=strongrule)
    output_beta = np.zeros(len(_elim))
    output_beta[np.logical_not(_elim)] = _beta_md
    return _k, output_beta


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def solution_path_LM_strongrule(design_matrix, outcome, lambda_, beta_0 = np.ones(1), tol=1e-2, maxit=500, penalty="SCAD", a=3.7, gamma=2., add_intercept_column=True):
    '''
    Carry out the optimization for the solution path of a penalized LM with strong rule.
    '''
    #     add design matrix column for the intercept, if it's not there already
    _design_matrix = design_matrix.copy()
    if add_intercept_column == True:
        if np.any(design_matrix[:,0] != design_matrix[0,0]): # check if design matrix has included a column for intercept or not
            intercept_design = np.ones(N).reshape(-1, 1)
            _design_matrix = np.hstack((intercept_design, _design_matrix))
    beta_mat = np.empty((len(lambda_)+1, _design_matrix.shape[1]))
    beta_mat[0,:] = 0.
    _lambda_max = lambda_max_LM(_design_matrix, outcome)
    lambda_ = np.hstack((np.array([_lambda_max]), lambda_))
    elim = np.array([False]*_design_matrix.shape[1])
    for j in range(len(lambda_)-1):
        _elim = strong_rule_seq_LM(X=_design_matrix, y=outcome, beta_old=beta_mat[j,:], lambda_new=lambda_[j+1], lambda_old=lambda_[j])
        elim = np.logical_and(elim, _elim)
        _beta_0 = beta_mat[j,:]
        _new_beta = np.zeros(_design_matrix.shape[1])
        _new_beta[np.logical_not(elim)] = UAG_LM_SCAD_MCP(design_matrix=_design_matrix[:, np.logical_not(elim)], outcome=outcome, beta_0 = _beta_0[np.logical_not(elim)], tol=tol, maxit=maxit, _lambda=lambda_[j], penalty=penalty, a=a, gamma=gamma, add_intercept_column=False)[1]
        beta_mat[j+1,:] = _new_beta
    return beta_mat[1:,:]


##################################################################
########### LM AG SNP version using bed-reader ###################
##################################################################

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_convex_LM(N, SNP_ind, bed, beta_md, y, outcome_iid):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    p=len(list(bed.sid))
    gene_iid = np.array(list(bed.iid))
    _y = y[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
    gene_ind = np.intersect1d(gene_iid, outcome_iid, assume_unique=True, return_indices=True)[1]
    # first calcualte _=X@beta_md-y
    _ = np.zeros(N)
    for j in SNP_ind:
        _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
        _X = _X[gene_ind] # get gene iid also in outcome iid
        _ += _X*beta_md[j]
    _ -= _y
    # then calculate _XTXbeta = X.T@X@beta_md = X.T@_
    _XTXbeta = np.zeros(p)
    for j in SNP_ind:
        _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
        _X = _X[gene_ind] # get gene iid also in outcome iid
        _XTXbeta[j] = _X@_
    del _
    return 1/N*_XTXbeta

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_SCAD_LM(N, SNP_ind, bed, beta_md, y, outcome_iid, _lambda, a):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _SNP_update_smooth_grad_convex_LM(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid) + SCAD_concave_grad(x=beta_md, lambda_=_lambda, a=a)

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_MCP_LM(N, SNP_ind, bed, beta_md, y, outcome_iid, _lambda, gamma):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _SNP_update_smooth_grad_convex_LM(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid) + MCP_concave_grad(x=beta_md, lambda_=_lambda, gamma=gamma)


# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_lambda_max_LM(bed, y, outcome_iid, N, SNP_ind):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
#     X_temp = X.copy()
#     X_temp = X_temp[:,1:]
#     X_temp -= np.mean(X_temp,0).reshape(1,-1)
#     X_temp /= np.std(X_temp,0)
#     y_temp = y.copy()
#     y_temp -= np.mean(y)
#     y_temp /= np.std(y)
    p=len(list(bed.sid))
    grad_at_0 = _SNP_update_smooth_grad_convex_LM(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=np.zeros(p), y=y, outcome_iid=outcome_iid)
    return lambda_max



# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_UAG_LM_SCAD_MCP(bed_file, bim_file, fam_file, outcome, outcome_iid, SNP_ind, L_convex, beta_0 = np.ones(1), tol=1e-2, maxit=500, _lambda=.5, penalty="SCAD", a=3.7, gamma=2.):
    '''
    Carry out the optimization for penalized LM for a fixed lambda.
    '''
    bed = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    y = outcome
    p = len(list(bed.sid))
    gene_iid = np.array(list(bed.iid))
    N = len(np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1])
    if np.all(beta_0==np.ones(1)):
        _ = np.zeros(p)
        _y = y[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
        for j in SNP_ind:
            _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
            _X = _X[gene_ind] # get gene iid also in outcome iid
            _[j] = _X@_y
        beta = np.sign(_)
    else:
        beta = beta_0
    # passing other parameters
    smooth_grad = np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0
    
    if penalty == "SCAD":
#         L = np.max(np.array([L_convex, 1./(a-1)]))
        L = np.linalg.norm(np.array([L_convex, 1./(a-1)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = _SNP_update_smooth_grad_SCAD_LM(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, _lambda=_lambda, a=a)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    else:
#         L = np.max(np.array([L_convex, 1./(gamma)]))
        L = np.linalg.norm(np.array([L_convex, 1./(gamma)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = _SNP_update_smooth_grad_MCP_LM(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, _lambda=_lambda, gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    return k, beta_md


# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_solution_path_LM(bed_file, bim_file, fam_file, outcome, outcome_iid, lambda_, L_convex, SNP_ind, beta_0 = np.ones(1), tol=1e-2, maxit=500, penalty="SCAD", a=3.7, gamma=2.):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    bed = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    p = len(list(bed.sid))
    beta_mat = np.zeros((len(lambda_)+1, p))
    for j in range(len(lambda_)): 
        beta_mat[j+1,:] = SNP_UAG_LM_SCAD_MCP(bed_file=bed_file, bim_file=bim_file, fam_file=fam_file, outcome=outcome, SNP_ind=SNP_ind, L_convex=L_convex, beta_0 = beta_mat[j,:], tol=tol, maxit=maxit, _lambda=lambda_[j], penalty=penalty, outcome_iid=outcome_iid, a=a, gamma=gamma)[1]
    return beta_mat[1:,:]



###################################################################################
########### LM AG SNP version using bed-reader, multiprocess ######################
###################################################################################

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_convex_LM_parallel(N, SNP_ind, bed, beta_md, y, outcome_iid, chunck_size):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    p=len(list(bed.sid))
    gene_iid = np.array(list(bed.iid))
    _y = y[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
    gene_ind = np.intersect1d(gene_iid, outcome_iid, assume_unique=True, return_indices=True)[1]
    # first calcualte _=X@beta_md-y
    def __parallel_plus(_ind):
        import numpy as np
        __ = np.zeros(N)
        for j in _ind:
            _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
            _X = _X[gene_ind] # get gene iid also in outcome iid
            __ += _X*beta_md[j]
        return __
    # multiprocessing starts here
    n_slices = np.ceil(len(SNP_ind)/chunck_size)
    with mp.Pool(mp.cpu_count()) as pl:
        _ = pl.map(__parallel_plus, np.array_split(SNP_ind, n_slices))
    _ = np.array(_).sum(0)
    _ -= _y
    # then calculate _XTXbeta = X.T@X@beta_md = X.T@_
    def __parallel_assign(_ind):
        import numpy as np
        k=0
        __ = np.zeros(len(_ind))
        for j in _ind:
            _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
            _X = _X[gene_ind] # get gene iid also in outcome iid
            __[k] = _X@_
            k += 1
        return __
    # multiprocessing starts here
    n_slices = np.ceil(len(SNP_ind)/chunck_size)
    with mp.Pool(mp.cpu_count()) as pl:
        _XTXbeta = pl.map(__parallel_assign, np.array_split(SNP_ind, n_slices))
    __XTXbeta = np.hstack(_XTXbeta)
    _XTXbeta = np.zeros(p)
    _XTXbeta[SNP_ind] = __XTXbeta
    
    del _
    del __XTXbeta

    return 1/N*_XTXbeta

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_SCAD_LM_parallel(N, SNP_ind, bed, beta_md, y, outcome_iid, _lambda, a, chunck_size):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _SNP_update_smooth_grad_convex_LM_parallel(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, chunck_size=chunck_size) + SCAD_concave_grad(x=beta_md, lambda_=_lambda, a=a)

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_MCP_LM_parallel(N, SNP_ind, bed, beta_md, y, outcome_iid, _lambda, gamma, chunck_size):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _SNP_update_smooth_grad_convex_LM_parallel(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, chunck_size=chunck_size) + MCP_concave_grad(x=beta_md, lambda_=_lambda, gamma=gamma)


# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_lambda_max_LM_parallel(bed, y, outcome_iid, N, SNP_ind, chunck_size):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
#     X_temp = X.copy()
#     X_temp = X_temp[:,1:]
#     X_temp -= np.mean(X_temp,0).reshape(1,-1)
#     X_temp /= np.std(X_temp,0)
#     y_temp = y.copy()
#     y_temp -= np.mean(y)
#     y_temp /= np.std(y)
    grad_at_0 = _SNP_update_smooth_grad_convex_LM_parallel(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=np.zeros(len(SNP_ind)), y=y, outcome_iid=outcome_iid, chunck_size=chunck_size)
    return lambda_max



# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_UAG_LM_SCAD_MCP_parallel(bed_file, bim_file, fam_file, outcome, outcome_iid, SNP_ind, L_convex, beta_0 = np.ones(1), tol=1e-2, maxit=500, _lambda=.5, penalty="SCAD", a=3.7, gamma=2., chunck_size=50000):
    '''
    Carry out the optimization for penalized LM for a fixed lambda.
    '''
    bed = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    y = outcome
    p = len(list(bed.sid))
    gene_iid = np.array(list(bed.iid))
    N = len(np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1])
    if np.all(beta_0==np.ones(1)):        
        def __parallel_assign(_ind):
            import numpy as np
            k=0
            __ = np.zeros(len(_ind))
            for j in _ind:
                _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
                _X = _X[gene_ind] # get gene iid also in outcome iid
                __[k] = _X@y
                k += 1
            return __
        # multiprocessing starts here
        _y = y[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
        n_slices = np.ceil(len(SNP_ind)/chunck_size)
        with mp.Pool(mp.cpu_count()) as pl:
            _XTy = pl.map(__parallel_assign, np.array_split(SNP_ind, n_slices))
        _XTy = np.hstack(_XTy)
        beta = np.zeros(p)
        beta[SNP_ind] = np.sign(_XTy)
    else:
        beta = beta_0
    # passing other parameters
    smooth_grad = np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0
    
    if penalty == "SCAD":
#         L = np.max(np.array([L_convex, 1./(a-1)]))
        L = np.linalg.norm(np.array([L_convex, 1./(a-1)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = _SNP_update_smooth_grad_SCAD_LM_parallel(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, _lambda=_lambda, a=a, chunck_size=chunck_size)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    else:
#         L = np.max(np.array([L_convex, 1./(gamma)]))
        L = np.linalg.norm(np.array([L_convex, 1./(gamma)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = _SNP_update_smooth_grad_MCP_LM_parallel(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, _lambda=_lambda, gamma=gamma, chunck_size=chunck_size)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    return k, beta_md


# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_solution_path_LM_parallel(bed_file, bim_file, fam_file, outcome, outcome_iid, lambda_, L_convex, SNP_ind, beta_0 = np.ones(1), tol=1e-2, maxit=500, penalty="SCAD", a=3.7, gamma=2., chunck_size=50000):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    bed = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    p = len(list(bed.sid))
    beta_mat = np.zeros((len(lambda_)+1, p))
    for j in range(len(lambda_)): 
        beta_mat[j+1,:] = SNP_UAG_LM_SCAD_MCP_parallel(bed_file=bed_file, bim_file=bim_file, fam_file=fam_file, outcome=outcome, SNP_ind=SNP_ind, L_convex=L_convex, beta_0 = beta_mat[j,:], tol=tol, maxit=maxit, _lambda=lambda_[j], penalty=penalty, outcome_iid=outcome_iid, a=a, gamma=gamma, chunck_size=chunck_size)[1]
    return beta_mat[1:,:]



############################################################################################
#################### logistic normal memory version with numba #############################
############################################################################################

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def update_smooth_grad_convex_logistic(N, X, beta_md, y):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    return (X.T@(np.tanh(X@beta_md/2.)/2.-y+.5))/(2.*N)

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def update_smooth_grad_SCAD_logistic(N, X, beta_md, y, _lambda, a):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return update_smooth_grad_convex_logistic(N=N, X=X, beta_md=beta_md, y=y) + SCAD_concave_grad(x=beta_md, lambda_=_lambda, a=a)

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def update_smooth_grad_MCP_logistic(N, X, beta_md, y, _lambda, gamma):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return update_smooth_grad_convex_logistic(N=N, X=X, beta_md=beta_md, y=y) + MCP_concave_grad(x=beta_md, lambda_=_lambda, gamma=gamma)

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def eval_obj_SCAD_logistic(N, X, beta_md, y, _lambda, a, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X@x_temp
    return (error.T@error)/(2.*N) + np.sum(SCAD(x_temp, lambda_=_lambda, a=a))

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def eval_obj_MCP_logistic(N, X, beta_md, y, _lambda, gamma, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X@x_temp
    return (error.T@error)/(2*N) + np.sum(SCAD(x_temp, lambda_=_lambda, gamma=gamma))

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def lambda_max_logistic(X, y):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    grad_at_0 = (y-np.mean(y))@X_temp/(2*len(y))
    lambda_max = np.linalg.norm(grad_at_0[1:], ord=np.infty)
    return lambda_max

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def strong_rule_seq_logistic(X, y, beta_old, lambda_new, lambda_old):
    """
    Use sequential strong to determine which betas to be nullified next.
    """
    grad = np.abs((y-np.tanh(X@beta_old/2)/2-.5)@X_temp/(2*len(y)))
    eliminated = (grad < 2*lambda_new - lambda_old) # True means the value gets eliminated
    eliminated = np.hstack((np.array([False]), eliminated)) # because intercept coefficient is not penalized
    return eliminated

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def UAG_logistic_SCAD_MCP(design_matrix, outcome, beta_0 = np.ones(1), tol=1e-2, maxit=500, _lambda=.5, penalty="SCAD", a=3.7, gamma=2., L_convex=1.1, add_intercept_column = True):
    '''
    Carry out the optimization for penalized logistic model for a fixed lambda.
    '''
    X = design_matrix.copy()
    y = outcome.copy()
    N = X.shape[0]
    if np.all(beta_0==np.ones(1)):
        cov = (y - np.mean(y))@(X - 1/N*np.sum(X, 0).reshape(1,-1))
        beta = np.sign(cov)
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        if np.any(X[:,0] != X[0,0]): # check if design matrix has included a column for intercept or not
            intercept_design = np.ones(N).reshape(-1, 1)
            X = np.hstack((intercept_design, X))
            beta = np.hstack((np.array([0.]), beta))
    # passing other parameters
    p = X.shape[1] # so here p includes the intercept design matrix column 
    smooth_grad = np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
#     L_convex = 1/N*np.max(np.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1/N*(np.linalg.eigvalsh(X@X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0
    
    if penalty == "SCAD":
#         L = np.max(np.array([L_convex, 1./(a-1)]))
        L = np.linalg.norm(np.array([L_convex, 1./(a-1)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = update_smooth_grad_SCAD_logistic(N=N, X=X, beta_md=beta_md, y=y, _lambda=_lambda, a=a)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    else:
#         L = np.max(np.array([L_convex, 1./(gamma)]))
        L = np.linalg.norm(np.array([L_convex, 1./(gamma)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = update_smooth_grad_MCP_logistic(N=N, X=X, beta_md=beta_md, y=y, _lambda=_lambda, gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
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
#                 self.converged = np.all(np.max(np.abs(self.beta_md - self.old_beta)) < self.tol).item()
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
#                 self.converged = np.all(np.max(np.abs(self.beta_md - self.old_beta)) < self.tol).item()
#                 self.old_beta = self.beta_md.copy()
#                 self.eval_obj_MCP_logistic(self.beta_md, self.obj_value)
#                 self.eval_obj_MCP_logistic(self.beta, self.obj_value_ORIGINAL)
#                 self.eval_obj_MCP_logistic(self.beta_ag, self.obj_value_AG)
#             else:
#                 break
#     return self.report_results()


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def solution_path_logistic(design_matrix, outcome, lambda_, beta_0 = np.ones(1), tol=1e-2, maxit=500, penalty="SCAD", a=3.7, gamma=2., add_intercept_column=True):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    #     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        if np.any(X[:,0] != X[0,0]): # check if design matrix has included a column for intercept or not
            intercept_design = np.ones(N).reshape(-1, 1)
            _design_matrix = design_matrix.copy()
            _design_matrix = np.hstack((intercept_design, _design_matrix))
    beta_mat = np.zeros((len(lambda_)+1, _design_matrix.shape[1]))
    for j in range(len(lambda_)):
        beta_mat[j+1,:] = UAG_logistic_SCAD_MCP(design_matrix=_design_matrix, outcome=outcome, beta_0 = beta_mat[j,:], tol=tol, maxit=maxit, _lambda=lambda_[j], penalty=penalty, a=a, gamma=gamma, add_intercept_column=False)[1]
    return beta_mat[1:,:]



# with strong rule 

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _UAG_logistic_SCAD_MCP_strongrule(design_matrix, outcome, beta_0 = np.ones(1), tol=1e-2, maxit=500, _lambda=.5, penalty="SCAD", a=3.7, gamma=2., L_convex=1.1, add_intercept_column = True, strongrule=True):
    '''
    Carry out the optimization for a fixed lambda with strong rule.
    '''
    X = design_matrix.copy()
    y = outcome.copy()
    N = X.shape[0]
    if np.all(beta_0==np.ones(1)):
        cov = (y - np.mean(y))@(X - 1/N*np.sum(X, 0).reshape(1,-1))
        beta = np.sign(cov)
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        if np.any(X[:,0] != X[0,0]): # check if design matrix has included a column for intercept or not
            intercept_design = np.ones(N).reshape(-1, 1)
            X = np.hstack((intercept_design, X))
            beta = np.hstack((np.array([0.]), beta))
    if strongrule == True:
        _lambda_max = lambda_max_logistic(X, y)
        p_original = X.shape[1]
        elim = strong_rule_seq_logistic(X, y, beta_old=np.zeros(p_original), lambda_new=_lambda, lambda_old=_lambda_max)
        X = X[:, np.logical_not(elim)]
        beta = beta[np.logical_not(elim)]
        
    # passing other parameters
    p = X.shape[1] # so here p includes the intercept design matrix column 
    smooth_grad = np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
#     L_convex = 1/N*np.max(np.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1/N*(np.linalg.eigvalsh(X@X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0
    
    if penalty == "SCAD":
#         L = np.max(np.array([L_convex, 1./(a-1)]))
        L = np.linalg.norm(np.array([L_convex, 1./(a-1)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2./(1.+(1.+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1.-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = update_smooth_grad_SCAD_logistic(N=N, X=X, beta_md=beta_md, y=y, _lambda=_lambda, a=a)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    else:
#         L = np.max(np.array([L_convex, 1./(gamma)]))
        L = np.linalg.norm(np.array([L_convex, 1./(gamma)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1.+(1.+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1.-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = update_smooth_grad_MCP_logistic(N=N, X=X, beta_md=beta_md, y=y, _lambda=_lambda, gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
#     if strongrule == True:
#         _beta_output = np.zeros((p_original))
# #         _ = np.argwhere(np.logical_not(elim)).flatten()
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

@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def UAG_logistic_SCAD_MCP_strongrule(design_matrix, outcome, beta_0 = np.ones(1), tol=1e-2, maxit=500, _lambda=.5, penalty="SCAD", a=3.7, gamma=2., L_convex=1.1, add_intercept_column = True, strongrule=True):
    """
    Carry out the optimization for a fixed lambda for penanlized logistic model with strong rule.
    """
    _k, _beta_md, _elim = _UAG_logistic_SCAD_MCP_strongrule(design_matrix=design_matrix, outcome=outcome, beta_0 = beta_0, tol=tol, maxit=maxit, _lambda=_lambda, penalty=penalty, a=a, gamma=gamma, L_convex=L_convex, add_intercept_column = add_intercept_column, strongrule=strongrule)
    output_beta = np.zeros(len(_elim))
    output_beta[np.logical_not(_elim)] = _beta_md
    return _k, output_beta


@jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def solution_path_logistic_strongrule(design_matrix, outcome, lambda_, beta_0 = np.ones(1), tol=1e-2, maxit=500, penalty="SCAD", a=3.7, gamma=2., add_intercept_column=True):
    '''
    Carry out the optimization for the solution path of a penalized logistic model with strong rule.
    '''
    #     add design matrix column for the intercept, if it's not there already
    _design_matrix = design_matrix.copy()
    if add_intercept_column == True:
        if np.any(design_matrix[:,0] != design_matrix[0,0]): # check if design matrix has included a column for intercept or not
            intercept_design = np.ones(N).reshape(-1, 1)
            _design_matrix = np.hstack((intercept_design, _design_matrix))
    beta_mat = np.empty((len(lambda_)+1, _design_matrix.shape[1]))
    beta_mat[0,:] = 0.
    _lambda_max = lambda_max_logistic(_design_matrix, outcome)
    lambda_ = np.hstack((np.array([_lambda_max]), lambda_))
    elim = np.array([False]*_design_matrix.shape[1])
    for j in range(len(lambda_)-1):
        _elim = strong_rule_seq_logistic(X=_design_matrix, y=outcome, beta_old=beta_mat[j,:], lambda_new=lambda_[j+1], lambda_old=lambda_[j])
        elim = np.logical_and(elim, _elim)
        _beta_0 = beta_mat[j,:]
        _new_beta = np.zeros(_design_matrix.shape[1])
        _new_beta[np.logical_not(elim)] = UAG_logistic_SCAD_MCP(design_matrix=_design_matrix[:, np.logical_not(elim)], outcome=outcome, beta_0 = _beta_0[np.logical_not(elim)], tol=tol, maxit=maxit, _lambda=lambda_[j], penalty=penalty, a=a, gamma=gamma, add_intercept_column=False)[1]
        beta_mat[j+1,:] = _new_beta
    return beta_mat[1:,:]

############################################################################
############# logistic SNP version with bed-reader #########################
############################################################################

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_convex_logistic(N, SNP_ind, bed, beta_md, y, outcome_iid):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    p=len(list(bed.sid))
    gene_iid = np.array(list(bed.iid))
    _y = y[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
    gene_ind = np.intersect1d(gene_iid, outcome_iid, assume_unique=True, return_indices=True)[1]
    # first calcualte _=X@beta_md-_y
    _ = np.zeros(N)
    for j in SNP_ind:
        _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
        _X = _X[gene_ind] # get gene iid also in outcome iid
        _ += _X*beta_md[j]
    _ = np.tanh(_/2.)/2.-_y+.5
    # then calculate output
    _XTXbeta = np.zeros(p)
    for j in SNP_ind:
        _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
        _X = _X[gene_ind] # get gene iid also in outcome iid
        _XTXbeta[j] = _X@_
    del _
    return _XTXbeta/(2.*N)

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_SCAD_logistic(N, SNP_ind, bed, beta_md, y, outcome_iid, _lambda, a):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _SNP_update_smooth_grad_convex_logistic(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid) + SCAD_concave_grad(x=beta_md, lambda_=_lambda, a=a)

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_MCP_logistic(N, SNP_ind, bed, beta_md, y, outcome_iid, _lambda, gamma):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _SNP_update_smooth_grad_convex_logistic(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid) + MCP_concave_grad(x=beta_md, lambda_=_lambda, gamma=gamma)


# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_lambda_max_logistic(bed, y, outcome_iid, N, SNP_ind):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
#     X_temp = X.copy()
#     X_temp = X_temp[:,1:]
#     X_temp -= np.mean(X_temp,0).reshape(1,-1)
#     X_temp /= np.std(X_temp,0)
#     y_temp = y.copy()
#     y_temp -= np.mean(y)
#     y_temp /= np.std(y)
    p=len(list(bed.sid))
    grad_at_0 = _SNP_update_smooth_grad_convex_logistic(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=np.zeros(p), y=y, outcome_iid=outcome_iid)
    return lambda_max



# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_UAG_logistic_SCAD_MCP(bed_file, bim_file, fam_file, outcome, outcome_iid, SNP_ind, L_convex, beta_0 = np.ones(1), tol=1e-2, maxit=500, _lambda=.5, penalty="SCAD", a=3.7, gamma=2.):
    '''
    Carry out the optimization for penalized logistic for a fixed lambda.
    '''
    bed = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    y = outcome
    p = len(list(bed.sid))
    gene_iid = np.array(list(bed.iid))
    N = len(np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1])
    if np.all(beta_0==np.ones(1)):
        _ = np.zeros(p)
        _y = y[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
        for j in SNP_ind:
            _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
            _X = _X[gene_ind] # get gene iid also in outcome iid
            _[j] = _X@_y
        beta = np.sign(_)
    else:
        beta = beta_0
    # passing other parameters
    smooth_grad = np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0
    
    if penalty == "SCAD":
#         L = np.max(np.array([L_convex, 1./(a-1)]))
        L = np.linalg.norm(np.array([L_convex, 1./(a-1)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = _SNP_update_smooth_grad_SCAD_logistic(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, _lambda=_lambda, a=a)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    else:
#         L = np.max(np.array([L_convex, 1./(gamma)]))
        L = np.linalg.norm(np.array([L_convex, 1./(gamma)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = _SNP_update_smooth_grad_MCP_logistic(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, _lambda=_lambda, gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    return k, beta_md


# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_solution_path_logistic(bed_file, bim_file, fam_file, outcome, outcome_iid, lambda_, L_convex, SNP_ind, beta_0 = np.ones(1), tol=1e-2, maxit=500, penalty="SCAD", a=3.7, gamma=2.):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    bed = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    p = len(list(bed.sid))
    beta_mat = np.zeros((len(lambda_)+1, p))
    for j in range(len(lambda_)): 
        beta_mat[j+1,:] = SNP_UAG_logistic_SCAD_MCP(bed_file=bed_file, bim_file=bim_file, fam_file=fam_file, outcome=outcome, SNP_ind=SNP_ind, L_convex=L_convex, beta_0 = beta_mat[j,:], tol=tol, maxit=maxit, _lambda=lambda_[j], penalty=penalty, outcome_iid=outcome_iid, a=a, gamma=gamma)[1]
    return beta_mat[1:,:]


##############################################################################################
################ logsitic AG SNP bed-reader version with multiprocess ########################
##############################################################################################

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_convex_logistic_parallel(N, SNP_ind, bed, beta_md, y, outcome_iid, chunck_size):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    p=len(list(bed.sid))
    gene_iid = np.array(list(bed.iid))
    _y = y[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
    gene_ind = np.intersect1d(gene_iid, outcome_iid, assume_unique=True, return_indices=True)[1]
    # first calcualte _=X@beta_md-y
    def __parallel_plus(_ind):
        import numpy as np
        __ = np.zeros(N)
        for j in _ind:
            _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
            _X = _X[gene_ind] # get gene iid also in outcome iid
            __ += _X*beta_md[j]
        return __
    # multiprocessing starts here
    n_slices = np.ceil(len(SNP_ind)/chunck_size)
    with mp.Pool(mp.cpu_count()) as pl:
        _ = pl.map(__parallel_plus, np.array_split(SNP_ind, n_slices))
    _ = np.array(_).sum(0)
    _ = np.tanh(_/2.)/2.-_y+.5
    # then calculate _XTXbeta = X.T@X@beta_md = X.T@_
    def __parallel_assign(_ind):
        import numpy as np
        k=0
        __ = np.zeros(len(_ind))
        for j in _ind:
            _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
            _X = _X[gene_ind] # get gene iid also in outcome iid
            __[k] = _X@_
            k += 1
        return __
    # multiprocessing starts here
    n_slices = np.ceil(len(SNP_ind)/chunck_size)
    with mp.Pool(mp.cpu_count()) as pl:
        _XTXbeta = pl.map(__parallel_assign, np.array_split(SNP_ind, n_slices))
    __XTXbeta = np.hstack(_XTXbeta)
    _XTXbeta = np.zeros(p)
    _XTXbeta[SNP_ind] = __XTXbeta
    
    del _
    del __XTXbeta

    return 1/N*_XTXbeta

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_SCAD_logistic_parallel(N, SNP_ind, bed, beta_md, y, outcome_iid, _lambda, a, chunck_size):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _SNP_update_smooth_grad_convex_logistic_parallel(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, chunck_size=chunck_size) + SCAD_concave_grad(x=beta_md, lambda_=_lambda, a=a)

# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_MCP_logistic_parallel(N, SNP_ind, bed, beta_md, y, outcome_iid, _lambda, gamma, chunck_size):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _SNP_update_smooth_grad_convex_logistic_parallel(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, chunck_size=chunck_size) + MCP_concave_grad(x=beta_md, lambda_=_lambda, gamma=gamma)


# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_lambda_max_logistic_parallel(bed, y, outcome_iid, N, SNP_ind, chunck_size):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
#     X_temp = X.copy()
#     X_temp = X_temp[:,1:]
#     X_temp -= np.mean(X_temp,0).reshape(1,-1)
#     X_temp /= np.std(X_temp,0)
#     y_temp = y.copy()
#     y_temp -= np.mean(y)
#     y_temp /= np.std(y)
    grad_at_0 = _SNP_update_smooth_grad_convex_logistic_parallel(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=np.zeros(len(SNP_ind)), y=y, outcome_iid=outcome_iid, chunck_size=chunck_size)
    return lambda_max



# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_UAG_logistic_SCAD_MCP_parallel(bed_file, bim_file, fam_file, outcome, outcome_iid, SNP_ind, L_convex, beta_0 = np.ones(1), tol=1e-2, maxit=500, _lambda=.5, penalty="SCAD", a=3.7, gamma=2., chunck_size=50000):
    '''
    Carry out the optimization for penalized logistic for a fixed lambda.
    '''
    bed = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    y = outcome
    p = len(list(bed.sid))
    gene_iid = np.array(list(bed.iid))
    N = len(np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1])
    if np.all(beta_0==np.ones(1)):        
        def __parallel_assign(_ind):
            import numpy as np
            k=0
            __ = np.zeros(len(_ind))
            for j in _ind:
                _X = bed.read(np.s_[:,j], dtype=np.int8).flatten()
                _X = _X[gene_ind] # get gene iid also in outcome iid
                __[k] = _X@y
                k += 1
            return __
        # multiprocessing starts here
        _y = y[np.intersect1d(outcome_iid, gene_iid, assume_unique=True, return_indices=True)[1]]
        n_slices = np.ceil(len(SNP_ind)/chunck_size)
        with mp.Pool(mp.cpu_count()) as pl:
            _XTy = pl.map(__parallel_assign, np.array_split(SNP_ind, n_slices))
        _XTy = np.hstack(_XTy)
        beta = np.zeros(p)
        beta[SNP_ind] = np.sign(_XTy)
    else:
        beta = beta_0
    # passing other parameters
    smooth_grad = np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0
    
    if penalty == "SCAD":
#         L = np.max(np.array([L_convex, 1./(a-1)]))
        L = np.linalg.norm(np.array([L_convex, 1./(a-1)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = _SNP_update_smooth_grad_SCAD_logistic_parallel(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, _lambda=_lambda, a=a, chunck_size=chunck_size)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    else:
#         L = np.max(np.array([L_convex, 1./(gamma)]))
        L = np.linalg.norm(np.array([L_convex, 1./(gamma)]), ord=np.infty)
        opt_beta = .99/L
        while ((not converged) or (k<3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k>=3: # in this case, restart
                opt_alpha = 1. # restarting
                restart_k = k # restarting
            else: # restarting
                opt_alpha = 2/(1+(1+4./opt_alpha**2)**.5) #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound 
            opt_lambda = opt_beta/opt_alpha #parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            beta_md_old = beta_md.copy() # restarting
            beta_md = (1-opt_alpha)*beta_ag + opt_alpha*beta
            old_speed_norm = speed_norm # restarting
            speed_norm = np.linalg.norm(beta_md - beta_md_old, ord=2) # restarting
            converged = (np.linalg.norm(beta_md - beta_md_old, ord=np.infty) < tol)
            smooth_grad = _SNP_update_smooth_grad_MCP_logistic_parallel(N=N, SNP_ind=SNP_ind, bed=bed, beta_md=beta_md, y=y, outcome_iid=outcome_iid, _lambda=_lambda, gamma=gamma, chunck_size=chunck_size)
            beta = soft_thresholding(x=beta - opt_lambda*smooth_grad, lambda_=opt_lambda*_lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta*smooth_grad, lambda_=opt_beta*_lambda)
#             converged = np.all(np.max(np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (np.linalg.norm(beta_md - beta_ag, ord=np.infty) < (tol*opt_beta))
    return k, beta_md


# @jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_solution_path_logistic_parallel(bed_file, bim_file, fam_file, outcome, outcome_iid, lambda_, L_convex, SNP_ind, beta_0 = np.ones(1), tol=1e-2, maxit=500, penalty="SCAD", a=3.7, gamma=2., chunck_size=50000):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    bed = open_bed(filepath=bed_file, fam_filepath=fam_file, bim_filepath=bim_file)
    p = len(list(bed.sid))
    beta_mat = np.zeros((len(lambda_)+1, p))
    for j in range(len(lambda_)): 
        beta_mat[j+1,:] = SNP_UAG_logistic_SCAD_MCP_parallel(bed_file=bed_file, bim_file=bim_file, fam_file=fam_file, outcome=outcome, SNP_ind=SNP_ind, L_convex=L_convex, beta_0 = beta_mat[j,:], tol=tol, maxit=maxit, _lambda=lambda_[j], penalty=penalty, outcome_iid=outcome_iid, a=a, gamma=gamma, chunck_size=chunck_size)[1]
    return beta_mat[1:,:]





