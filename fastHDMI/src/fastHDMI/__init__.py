#!/usr/bin/env python
# coding: utf-8

import warnings as _warnings
import multiprocess as _mp
import ctypes as _ctypes
from sklearn.preprocessing import RobustScaler as _scaler
from sklearn.decomposition import PCA as _PCA
from KDEpy.bw_selection import silvermans_rule as _silvermans_rule
from KDEpy.bw_selection import improved_sheather_jones as _improved_sheather_jones
from KDEpy.bw_selection import scotts_rule as _scotts_rule
from sklearn.feature_selection import mutual_info_regression as _mutual_info_regression
from sklearn.feature_selection import mutual_info_classif as _mutual_info_classif
from dask import dataframe as _dd
import pandas as _pd
from KDEpy import FFTKDE as _FFTKDE
# from bed_reader import open_bed as _open_bed
from numba import njit as _njit
from numba import jit as _jit
import numpy as _np
from tqdm import tqdm as _tqdm

# if the CPU supports AVX2, use AVX2; otherwise, nvm
import os


def supports_avx2():
    # Using a simple way to check for AVX2 support
    return "avx2" in os.popen("cat /proc/cpuinfo").read()


if supports_avx2():
    from fastHDMI.cython_fun import joint_to_mi_cython, hist_obj_cython, num_of_bins_cython, binning_MI_cython, binning_MI_discrete_cython
else:
    from fastHDMI.cython_fun_notusingavx2 import joint_to_mi_cython, hist_obj_cython, num_of_bins_cython, binning_MI_cython, binning_MI_discrete_cython

_warnings.filterwarnings('ignore')


#############################################################################
############# clumping and screening using mutual information ###############
#############################################################################
def _open_bed():
    print(
        "bed_reader might have some bugs for this version, causing it not running"
    )
    pass


def _hist_obj(x, D):
    """
    For a given 1-D array x and given number of bins D, calculate the value of the objective function based on the method given in "How many bins should be put in a regular histogram" by Birge and Rozenholc.
    """
    N_j, _ = _np.histogram(
        x, bins=D)  # to get the number of data points in each bin
    return _np.sum(
        N_j * _np.log(N_j)) + len(x) * _np.log(D) - (D - 1 + _np.log(D)**2.5)


def _num_of_bins(x):
    """
    For a given 1-D array x, calculate the number of bins based on the method given in "How many bins should be put in a regular histogram" by Birge and Rozenholc.
    """
    D_list = _np.arange(
        2, 100)  # search for the optimal number of bins from 2 to 100
    D_obj_list = _np.array([hist_obj_cython(x, D) for D in D_list])
    return D_list[_np.nanargmax(D_obj_list)]


def _binning_MI(a, b):
    """
    For two 1-D arrays continuous a, b; calculate their mutual information using binning.
    """
    joint, _, _ = _np.histogram2d(a,
                                  b,
                                  bins=(_num_of_bins(a), _num_of_bins(b)))
    joint /= _np.sum(joint)
    # Convert joint to a contiguous array for performance
    joint = _np.ascontiguousarray(joint)
    return joint_to_mi_cython(joint)


def _binning_MI_discrete(a, b):
    """
    For two 1-D arrays discrete a and continuous b; calculate their mutual information using binning.
    """
    joint, _, _ = _np.histogram2d(a,
                                  b,
                                  bins=(len(_np.unique(a)), _num_of_bins(b)))
    joint /= _np.sum(joint)
    # Convert joint to a contiguous array for performance
    joint = _np.ascontiguousarray(joint)
    return joint_to_mi_cython(joint)


@_njit(cache=True)
def _nan_inf_to_0(x):
    """
    Convert NaN and infinity values in a NumPy array to zero.
    """
    return _np.where(_np.isfinite(x), x, 0.)


@_njit(cache=True)
def _compute_log_marginals(joint, forward_euler_a, forward_euler_b):
    """
    Compute log marginals from a joint distribution.
    """
    log_marginal_x = _np.log(_np.sum(joint, axis=1)) + _np.log(forward_euler_b)
    log_marginal_y = _np.log(_np.sum(joint, axis=0)) + _np.log(forward_euler_a)
    return _nan_inf_to_0(log_marginal_x), _nan_inf_to_0(log_marginal_y)


@_njit(cache=True)
def _joint_to_mi(joint, forward_euler_a=1., forward_euler_b=1.):
    """
    Calculate mutual information from a joint distribution.
    """
    joint /= _np.sum(joint) * forward_euler_a * forward_euler_b
    log_marginal_a, log_marginal_b = _compute_log_marginals(
        joint, forward_euler_a, forward_euler_b)
    log_joint = _nan_inf_to_0(_np.log(joint))
    mi_temp = _np.sum(
        joint *
        (log_joint - log_marginal_a.reshape(-1, 1) -
         log_marginal_b.reshape(1, -1))) * forward_euler_a * forward_euler_b
    return max(mi_temp, 0.0)


def _select_bandwidth(input_var, bw_multiplier, bw="silverman"):
    """
    Select bandwidth for univariate data.
    """
    bandwidth_functions = {
        "silverman": _silvermans_rule,
        "scott": _scotts_rule,
        "ISJ": _improved_sheather_jones
    }

    if isinstance(bw, str):
        _bw = bandwidth_functions[bw](input_var.reshape(-1, 1))
    elif isinstance(bw, float):
        _bw = bw
    else:
        raise ValueError("Invalid bandwidth selection method.")

    return _bw * bw_multiplier


def _univariate_bw(input_var, bw_multiplier, bw="silverman"):
    """
    Output bandwidth from univariate data.
    """
    return _select_bandwidth(input_var, bw_multiplier, bw)


def _bivariate_bw(_data, bw_multiplier, bw="silverman"):
    """
    Output bandwidth from bivariate data.
    """
    if isinstance(bw, str):
        bw1 = _select_bandwidth(_data[:, [0]], bw_multiplier, bw)
        bw2 = _select_bandwidth(_data[:, [1]], bw_multiplier, bw)
    elif isinstance(bw, (_np.ndarray, list)) and len(bw) == 2:
        bw1, bw2 = bw[0], bw[1]
    else:
        raise ValueError("Invalid bandwidth selection for bivariate data.")

    return bw1 * bw_multiplier, bw2 * bw_multiplier


def MI_continuous_012(a,
                      b,
                      bw_multiplier,
                      N=500,
                      kernel="epa",
                      bw="silverman",
                      **kwarg):
    """
    Calculate mutual information between a continuous outcome and a SNP variable of 0, 1, 2, or a binary variable.
    Assumes no missing data.
    """
    # Calculate the probabilities for each SNP value
    p0 = _np.count_nonzero(b == 0) / len(b)
    p1 = _np.count_nonzero(b == 1) / len(b)
    p2 = 1. - p0 - p1

    # Standardize 'a'
    _a = _scaler().fit_transform(a.reshape(-1, 1)).flatten()

    # Get the boundary width for the joint density grid
    _bw = _univariate_bw(_a, bw_multiplier, bw)
    a_temp, _ = _FFTKDE(kernel=kernel, bw=_bw,
                        **kwarg).fit(data=_a).evaluate(N)

    # Initialize joint distribution array
    joint = _np.zeros((N, 3))

    # Calculate conditional densities for each SNP value
    for i, (p,
            condition) in enumerate(zip([p0, p1, p2],
                                        [b == 0, b == 1, b == 2])):
        if _np.count_nonzero(condition) > 2:
            _bw = _univariate_bw(_a[condition], bw_multiplier, bw)
            kde = _FFTKDE(kernel=kernel, bw=_bw,
                          **kwarg).fit(data=_a[condition])
            joint[:, i] = kde.evaluate(a_temp)[0] * p
        # No else block needed; joint[:, i] is already initialized with zeros

    # Calculate the forward Euler step
    forward_euler_step = a_temp[1] - a_temp[0]

    # Ensure all values in joint are non-negative
    joint = _np.clip(joint, 0, None)

    # Convert joint to a contiguous array for performance
    joint = _np.ascontiguousarray(joint)

    return joint_to_mi_cython(joint=joint, forward_euler_a=forward_euler_step)

    return mi_temp


def MI_binary_012(a, b):
    """
    Calculate mutual information between a binary outcome and a SNP variable (0, 1, 2).
    Assumes no missing data.
    """
    return MI_012_012(a, b)


def MI_012_012(a, b):
    """
    Calculate mutual information between two SNP variables (each variable can be 0, 1, 2).
    Assumes no missing data. Useful for MI-based clumping.
    """
    joint = _np.array([[(_np.logical_and(a == i, b == j)).sum() / len(a)
                        for j in range(3)] for i in range(3)])
    return joint_to_mi_cython(joint=_np.ascontiguousarray(joint))


def MI_continuous_continuous(a,
                             b,
                             bw_multiplier,
                             a_N=300,
                             b_N=300,
                             kernel="epa",
                             bw="silverman",
                             norm=2,
                             **kwarg):
    """
    Calculate mutual information between two continuous random variables.
    """
    sorted_indices = _np.argsort(a)
    data = _np.vstack((a[sorted_indices], b[sorted_indices])).T
    _data = _scaler().fit_transform(data)

    bw1, bw2 = _bivariate_bw(_data, bw_multiplier, bw)
    grid, joint = _FFTKDE(kernel=kernel, norm=norm,
                          **kwarg).fit(_data).evaluate((a_N, b_N))
    joint = joint.reshape(b_N, -1).T
    joint[joint < 0] = 0

    a_step, b_step = grid[b_N, 0] - grid[0, 0], grid[1, 1] - grid[0, 1]
    return joint_to_mi_cython(joint=_np.ascontiguousarray(joint),
                              forward_euler_a=a_step,
                              forward_euler_b=b_step)


def MI_binary_continuous(a,
                         b,
                         bw_multiplier,
                         N=500,
                         kernel="epa",
                         bw="silverman",
                         **kwarg):
    """
    Calculate mutual information between a binary and a continuous variable.
    """
    return MI_continuous_012(a=b,
                             b=a,
                             N=N,
                             kernel=kernel,
                             bw=bw,
                             bw_multiplier=bw_multiplier,
                             **kwarg)


@_njit(cache=True)
def Pearson_to_MI_Gaussian(corr):
    """
    Convert Pearson correlation coefficient to mutual information for bivariate Gaussian variables.

    Parameters:
    corr (float): Pearson correlation coefficient.

    Returns:
    float: Mutual information.
    """
    if corr == -1 or corr == 1:
        return _np.inf
    return -0.5 * (_np.log1p(-corr**2))


@_njit(cache=True)
def MI_to_Linfoot(mi):
    """
    Convert mutual information to Linfoot's measure of association.

    Parameters:
    mi (float): Mutual information.

    Returns:
    float: Linfoot's measure of association.
    """
    if mi < 0:
        raise ValueError("Mutual information cannot be negative.")
    return (1. - _np.exp(-2. * mi))**0.5


# outcome_iid should be a  list of strings for identifiers
def continuous_screening_plink(bed_file,
                               bim_file,
                               fam_file,
                               outcome,
                               outcome_iid,
                               N=500,
                               kernel="epa",
                               bw="silverman",
                               bw_multiplier=1.,
                               verbose=1,
                               **kwarg):
    """
    (Single Core version) take plink files to calculate the mutual information between the continuous outcome and many SNP variables.
    """
    bed1 = _open_bed(filepath=bed_file,
                     fam_filepath=fam_file,
                     bim_filepath=bim_file)
    gene_iid = _np.array(list(bed1.iid))
    bed1_sid = _np.array(list(bed1.sid))
    outcome = outcome[_np.intersect1d(outcome_iid,
                                      gene_iid,
                                      assume_unique=True,
                                      return_indices=True)[1]]

    # get genetic indices
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]

    def _map_foo(j):
        _SNP = bed1.read(_np.s_[:, j], dtype=_np.int8).flatten()
        _SNP = _SNP[gene_ind]  # get gene iid also in outcome iid
        _outcome = outcome[_SNP != -127]  # remove missing SNP in outcome
        _SNP = _SNP[_SNP != -127]  # remove missing SNP
        return MI_continuous_012(a=_outcome,
                                 b=_SNP,
                                 N=N,
                                 kernel=kernel,
                                 bw=bw,
                                 bw_multiplier=bw_multiplier,
                                 **kwarg)

    _iter = range(len(bed1_sid))
    if verbose > 1:
        _iter = _tqdm(iter)
    MI_UKBB = _np.array(list(map(_map_foo, _iter)))
    return MI_UKBB


def binary_screening_plink(bed_file,
                           bim_file,
                           fam_file,
                           outcome,
                           outcome_iid,
                           verbose=1,
                           **kwarg):
    """
    (Single Core version) take plink files to calculate the mutual information between the binary outcome and many SNP variables.
    """
    bed1 = _open_bed(filepath=bed_file,
                     fam_filepath=fam_file,
                     bim_filepath=bim_file)
    gene_iid = _np.array(list(bed1.iid))
    bed1_sid = _np.array(list(bed1.sid))
    outcome = outcome[_np.intersect1d(outcome_iid,
                                      gene_iid,
                                      assume_unique=True,
                                      return_indices=True)[1]]
    # get genetic indices
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]

    def _map_foo(j):
        _SNP = bed1.read(_np.s_[:, j], dtype=_np.int8).flatten()
        _SNP = _SNP[gene_ind]  # get gene iid also in outcome iid
        _outcome = outcome[_SNP != -127]  # remove missing SNP in outcome
        _SNP = _SNP[_SNP != -127]  # remove missing SNP
        return MI_binary_012(a=_outcome, b=_SNP, **kwarg)

    _iter = range(len(bed1_sid))
    if verbose >= 1:
        _iter = _tqdm(_iter)
    MI_UKBB = _np.array(list(map(_map_foo, _iter)))
    return MI_UKBB


def continuous_screening_plink_parallel(bed_file,
                                        bim_file,
                                        fam_file,
                                        outcome,
                                        outcome_iid,
                                        N=500,
                                        kernel="epa",
                                        bw="silverman",
                                        bw_multiplier=1.,
                                        core_num="NOT DECLARED",
                                        multp=10,
                                        verbose=1,
                                        **kwarg):
    """
    (Multiprocessing version) take plink files to calculate the mutual information between the continuous outcome and many SNP variables.
    """
    # check some basic things
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    # read some metadata
    bed1 = _open_bed(filepath=bed_file,
                     fam_filepath=fam_file,
                     bim_filepath=bim_file)
    gene_iid = _np.array(list(bed1.iid))
    bed1_sid = _np.array(list(bed1.sid))
    outcome = outcome[_np.intersect1d(outcome_iid,
                                      gene_iid,
                                      assume_unique=True,
                                      return_indices=True)[1]]
    # get genetic indices
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]

    def _continuous_screening_plink_slice(_slice):

        def _map_foo(j):
            _SNP = bed1.read(_np.s_[:, j], dtype=_np.int8).flatten()
            _SNP = _SNP[gene_ind]  # get gene iid also in outcome iid
            _outcome = outcome[_SNP != -127]  # remove missing SNP in outcome
            _SNP = _SNP[_SNP != -127]  # remove missing SNP
            return MI_continuous_012(a=_outcome,
                                     b=_SNP,
                                     N=N,
                                     kernel=kernel,
                                     bw=bw,
                                     bw_multiplier=bw_multiplier,
                                     **kwarg)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(len(bed1_sid))
    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_UKBB = pl.map(_continuous_screening_plink_slice, _iter)
    MI_UKBB = _np.hstack(MI_UKBB)
    return MI_UKBB


def binary_screening_plink_parallel(bed_file,
                                    bim_file,
                                    fam_file,
                                    outcome,
                                    outcome_iid,
                                    core_num="NOT DECLARED",
                                    multp=10,
                                    verbose=1,
                                    **kwarg):
    """
    (Multiprocessing version) take plink files to calculate the mutual information between the binary outcome and many SNP variables.
    """
    # check basic things
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    # read some metadata
    bed1 = _open_bed(filepath=bed_file,
                     fam_filepath=fam_file,
                     bim_filepath=bim_file)
    gene_iid = _np.array(list(bed1.iid))
    bed1_sid = _np.array(list(bed1.sid))
    outcome = outcome[_np.intersect1d(outcome_iid,
                                      gene_iid,
                                      assume_unique=True,
                                      return_indices=True)[1]]
    # get genetic indices
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]

    def _binary_screening_plink_slice(_slice):

        def _map_foo(j):
            _SNP = bed1.read(_np.s_[:, j], dtype=_np.int8).flatten()
            _SNP = _SNP[gene_ind]  # get gene iid also in outcome iid
            _outcome = outcome[_SNP != -127]  # remove missing SNP in outcome
            _SNP = _SNP[_SNP != -127]  # remove missing SNP
            return MI_binary_012(a=_outcome, b=_SNP, **kwarg)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(len(bed1_sid))
    _iter = _np.array_split(ind, core_num * multp)
    if verbose > 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_UKBB = pl.map(_binary_screening_plink_slice, _iter)
    MI_UKBB = _np.hstack(MI_UKBB)
    return MI_UKBB


def clump_plink_parallel(bed_file,
                         bim_file,
                         fam_file,
                         clumping_threshold=Pearson_to_MI_Gaussian(.6),
                         num_SNPS_exam=_np.infty,
                         core_num="NOT DECLARED",
                         multp=10,
                         verbose=1):
    """
    (Multiprocessing version) take plink files to calculate the mutual information between the binary outcome and many SNP variables.
    """
    # check basic things
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    # read some metadata
    bed1 = _open_bed(filepath=bed_file,
                     fam_filepath=fam_file,
                     bim_filepath=bim_file)
    bed1_sid = _np.array(list(bed1.sid))
    if num_SNPS_exam == _np.infty:
        num_SNPS_exam = len(bed1_sid) - 1
    keep_cols = _np.arange(
        len(bed1_sid))  # pruning by keeping all SNPS at the beginning
    _iter = _np.arange(num_SNPS_exam)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    for current_var_ind in _iter:  # note that here _iter and keep_cols don't need to agree, by the break command comes later
        if current_var_ind + 1 <= len(keep_cols):
            outcome = bed1.read(_np.s_[:, current_var_ind],
                                dtype=_np.int8).flatten()
            gene_ind = _np.where(outcome != -127)
            outcome = outcome[gene_ind]

            def _012_012_plink_slice(_slice):

                def _map_foo(j):
                    _SNP = bed1.read(_np.s_[:, j], dtype=_np.int8).flatten()
                    _SNP = _SNP[gene_ind]  # get gene iid also in outcome iid
                    _outcome = outcome[_SNP !=
                                       -127]  # remove missing SNP in outcome
                    _SNP = _SNP[_SNP != -127]  # remove missing SNP
                    return MI_012_012(a=_outcome, b=_SNP)

                _MI_slice = _np.array(list(map(_map_foo, _slice)))
                return _MI_slice

            # multiprocessing starts here
            ind = keep_cols[current_var_ind + 1:]
            __iter = _np.array_split(ind, core_num * multp)
            with _mp.Pool(core_num) as pl:
                MI_UKBB = pl.map(_012_012_plink_slice, __iter)
            MI_UKBB = _np.hstack(MI_UKBB)
            keep_cols = _np.hstack(
                (keep_cols[:current_var_ind + 1],
                 keep_cols[current_var_ind +
                           1:][MI_UKBB <= clumping_threshold]))
        else:
            break
    return current_var_ind, bed1_sid[keep_cols]


def _read_csv(csv_file, _usecols, csv_engine, parquet_file, sample, verbose=1):
    """
    Read a csv file using differnet engines. Use dask to read csv if low in memory.
    """
    assert csv_engine in [
        "dask", "pyarrow", "fastparquet", "c", "python"
    ], "Only dask and pandas csv engines or fastparquet are supported to read csv files."
    if _np.array(_usecols).size == 0:
        if verbose > 1:
            print(
                "Variable names not provided -- start reading variable names from csv file now, might take some time, depending on the csv file size."
            )
        if csv_engine == "dask":
            _df = _dd.read_csv(csv_file, sample=sample)
            _usecols = _np.array(list(_df.columns)[1:])
        elif csv_engine in ["pyarrow", "c",
                            "python"]:  # these are pandas CSV engines
            _df = _pd.read_csv(csv_file,
                               encoding='unicode_escape',
                               engine=csv_engine)
            _usecols = _np.array(_df.columns.to_list()[1:])
        elif csv_engine == "fastparquet":
            _df = _pd.read_parquet(parquet_file, engine="fastparquet")
            _usecols = _np.array(_df.columns.to_list()[1:])
        if verbose > 1:
            print("Reading variable names from csv file finished.")
    else:
        _usecols = _np.array(_usecols)
        if csv_engine == "dask":
            _df = _dd.read_csv(csv_file, names=_usecols, sample=sample)
        elif csv_engine in ["pyarrow", "c", "python"]:
            _df = _pd.read_csv(csv_file,
                               encoding='unicode_escape',
                               usecols=_usecols,
                               engine=csv_engine)
        elif csv_engine == "fastparquet":
            _df = _pd.read_parquet(parquet_file,
                                   engine="fastparquet")[_usecols]
    return _df, _usecols


def _read_two_columns(_df, __, csv_engine):
    """
    Read two columns from a dataframe object, remove NaN. Use dask to read csv if low in memory.
    """
    if csv_engine == "dask":
        _ = _np.asarray(_df[__].dropna().compute())
    elif csv_engine in ["pyarrow", "c", "python",
                        "fastparquet"]:  # these are engines using pandas
        _ = _df[__].dropna().to_numpy()

    _a = _[:, 0].copy()  # such that _df won't be mutated
    _b = _[:, 1].copy()  # such that _df won't be mutated
    return _a, _b


def binary_screening_csv(csv_file="_",
                         _usecols=[],
                         N=500,
                         kernel="epa",
                         bw="silverman",
                         bw_multiplier=1.,
                         csv_engine="c",
                         parquet_file="_",
                         sample=256000,
                         verbose=1,
                         **kwarg):
    """
    Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    The outcome should be binary and the covariates be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    """
    assert csv_file != "_" or parquet_file != "_", "CSV or parquet filepath should be declared"
    # outcome is the first variable by default; if other specifications are needed, put it the first item in _usecols
    # read csv
    _df, _usecols = _read_csv(csv_file=csv_file,
                              _usecols=_usecols,
                              csv_engine=csv_engine,
                              parquet_file=parquet_file,
                              sample=sample,
                              verbose=verbose)

    def _map_foo(j):
        __ = [
            _usecols[0], _usecols[j + 1]
        ]  # here using _usecol[j + 1] because the left first column is the outcome
        _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
        return MI_binary_continuous(a=_a,
                                    b=_b,
                                    N=N,
                                    kernel=kernel,
                                    bw=bw,
                                    bw_multiplier=bw_multiplier,
                                    **kwarg)

    _iter = _np.arange(len(_usecols) - 1)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    MI_df = _np.array(list(map(_map_foo, _iter)))

    del _df

    return MI_df


def continuous_screening_csv(csv_file="_",
                             _usecols=[],
                             a_N=300,
                             b_N=300,
                             kernel="epa",
                             bw="silverman",
                             bw_multiplier=1.,
                             norm=2,
                             csv_engine="c",
                             parquet_file="_",
                             sample=256000,
                             verbose=1,
                             **kwarg):
    """
    Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    Both the outcome and the covariates should be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    """
    assert csv_file != "_" or parquet_file != "_", "CSV or parquet filepath should be declared"
    # read csv
    _df, _usecols = _read_csv(csv_file=csv_file,
                              _usecols=_usecols,
                              csv_engine=csv_engine,
                              parquet_file=parquet_file,
                              sample=sample,
                              verbose=verbose)

    def _map_foo(j):
        __ = [
            _usecols[0], _usecols[j + 1]
        ]  # here using _usecol[j + 1] because the left first column is the outcome
        _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
        return MI_continuous_continuous(a=_a,
                                        b=_b,
                                        a_N=a_N,
                                        b_N=b_N,
                                        kernel=kernel,
                                        bw=bw,
                                        bw_multiplier=bw_multiplier,
                                        norm=norm,
                                        **kwarg)

    _iter = _np.arange(len(_usecols) - 1)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    MI_df = _np.array(list(map(_map_foo, _iter)))

    del _df

    return MI_df


def binary_screening_csv_parallel(csv_file="_",
                                  _usecols=[],
                                  N=500,
                                  kernel="epa",
                                  bw="silverman",
                                  bw_multiplier=1.,
                                  core_num="NOT DECLARED",
                                  multp=10,
                                  csv_engine="c",
                                  parquet_file="_",
                                  sample=256000,
                                  verbose=1,
                                  share_memory=True,
                                  **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    The outcome should be binary and the covariates be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    # check some basic things
    assert csv_file != "_" or parquet_file != "_", "CSV or parquet filepath should be declared"

    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    # read csv
    _df, _usecols = _read_csv(csv_file=csv_file,
                              _usecols=_usecols,
                              csv_engine=csv_engine,
                              parquet_file=parquet_file,
                              sample=sample,
                              verbose=verbose)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _binary_screening_csv_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return MI_binary_continuous(a=_a,
                                        b=_b,
                                        N=N,
                                        kernel=kernel,
                                        bw=bw,
                                        bw_multiplier=bw_multiplier,
                                        **kwarg)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here

    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome
    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_binary_screening_csv_slice, _iter)
    MI_df = _np.hstack(MI_df)

    del _df

    return MI_df


def continuous_screening_csv_parallel(csv_file="_",
                                      _usecols=[],
                                      a_N=300,
                                      b_N=300,
                                      kernel="epa",
                                      bw="silverman",
                                      bw_multiplier=1.,
                                      norm=2,
                                      core_num="NOT DECLARED",
                                      multp=10,
                                      csv_engine="c",
                                      parquet_file="_",
                                      sample=256000,
                                      verbose=1,
                                      share_memory=True,
                                      **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    Both the outcome and the covariates should be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    # check some basic things
    assert csv_file != "_" or parquet_file != "_", "CSV or parquet filepath should be declared"

    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    # read csv
    _df, _usecols = _read_csv(csv_file=csv_file,
                              _usecols=_usecols,
                              csv_engine=csv_engine,
                              parquet_file=parquet_file,
                              sample=sample,
                              verbose=verbose)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _continuous_screening_csv_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return MI_continuous_continuous(a=_a,
                                            b=_b,
                                            a_N=a_N,
                                            b_N=b_N,
                                            kernel=kernel,
                                            bw=bw,
                                            bw_multiplier=bw_multiplier,
                                            norm=norm,
                                            **kwarg)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome

    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_continuous_screening_csv_slice, _iter)
    MI_df = _np.hstack(MI_df)

    del _df

    return MI_df


def binning_binary_screening_csv_parallel(csv_file="_",
                                          _usecols=[],
                                          core_num="NOT DECLARED",
                                          multp=10,
                                          csv_engine="c",
                                          parquet_file="_",
                                          sample=256000,
                                          verbose=1,
                                          share_memory=True,
                                          **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    The outcome should be binary and the covariates be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    # check some basic things
    assert csv_file != "_" or parquet_file != "_", "CSV or parquet filepath should be declared"

    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    # read csv
    _df, _usecols = _read_csv(csv_file=csv_file,
                              _usecols=_usecols,
                              csv_engine=csv_engine,
                              parquet_file=parquet_file,
                              sample=sample,
                              verbose=verbose)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _binary_screening_csv_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            _a = _a.astype(
                float
            )  # recall our binning_MI_discrete_cython doesn't accept int data type
            return binning_MI_discrete_cython(a=_a, b=_b)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here

    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome
    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_binary_screening_csv_slice, _iter)
    MI_df = _np.hstack(MI_df)

    del _df

    return MI_df


def binning_continuous_screening_csv_parallel(csv_file="_",
                                              _usecols=[],
                                              core_num="NOT DECLARED",
                                              multp=10,
                                              csv_engine="c",
                                              parquet_file="_",
                                              sample=256000,
                                              verbose=1,
                                              share_memory=True,
                                              **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    Both the outcome and the covariates should be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    # check some basic things
    assert csv_file != "_" or parquet_file != "_", "CSV or parquet filepath should be declared"

    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    # read csv
    _df, _usecols = _read_csv(csv_file=csv_file,
                              _usecols=_usecols,
                              csv_engine=csv_engine,
                              parquet_file=parquet_file,
                              sample=sample,
                              verbose=verbose)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _continuous_screening_csv_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return binning_MI_cython(a=_a, b=_b)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome

    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_continuous_screening_csv_slice, _iter)
    MI_df = _np.hstack(MI_df)

    del _df

    return MI_df


def binary_skMI_screening_csv_parallel(csv_file="_",
                                       _usecols=[],
                                       n_neighbors=3,
                                       core_num="NOT DECLARED",
                                       multp=10,
                                       csv_engine="c",
                                       parquet_file="_",
                                       sample=256000,
                                       verbose=1,
                                       share_memory=True,
                                       **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    Both the outcome and the covariates should be binary. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    # check some basic things
    assert csv_file != "_" or parquet_file != "_", "CSV or parquet filepath should be declared"

    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    # read csv
    _df, _usecols = _read_csv(csv_file=csv_file,
                              _usecols=_usecols,
                              csv_engine=csv_engine,
                              parquet_file=parquet_file,
                              sample=sample,
                              verbose=verbose)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _binary_skMI_df_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return _mutual_info_classif(y=_a.reshape(-1, 1),
                                        X=_b.reshape(-1, 1),
                                        n_neighbors=n_neighbors,
                                        discrete_features=False,
                                        **kwarg)[0]

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome

    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_binary_skMI_df_slice, _iter)
    MI_df = _np.hstack(MI_df)

    del _df

    return MI_df


def continuous_skMI_screening_csv_parallel(csv_file="_",
                                           _usecols=[],
                                           n_neighbors=3,
                                           core_num="NOT DECLARED",
                                           multp=10,
                                           csv_engine="c",
                                           parquet_file="_",
                                           sample=256000,
                                           verbose=1,
                                           share_memory=True,
                                           **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    Both the outcome and the covariates should be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    # check some basic things
    assert csv_file != "_" or parquet_file != "_", "CSV or parquet filepath should be declared"

    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    # read csv
    _df, _usecols = _read_csv(csv_file=csv_file,
                              _usecols=_usecols,
                              csv_engine=csv_engine,
                              parquet_file=parquet_file,
                              sample=sample,
                              verbose=verbose)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _continuous_skMI_df_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return _mutual_info_regression(y=_a.reshape(-1, 1),
                                           X=_b.reshape(-1, 1),
                                           n_neighbors=n_neighbors,
                                           discrete_features=False,
                                           **kwarg)[0]

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome

    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_continuous_skMI_df_slice, _iter)
    MI_df = _np.hstack(MI_df)

    del _df

    return MI_df


def Pearson_screening_csv_parallel(csv_file="_",
                                   _usecols=[],
                                   core_num="NOT DECLARED",
                                   multp=10,
                                   csv_engine="c",
                                   parquet_file="_",
                                   sample=256000,
                                   verbose=1,
                                   share_memory=True):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the Pearson's correlation between outcome and covariates.
    If _usecols is given, the returned Pearson correlation will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    This function accounts for missing data better than the Pearson's correlation matrix function provided by numpy.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.    """
    # check some basic things
    assert csv_file != "_" or parquet_file != "_", "CSV or parquet filepath should be declared"

    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    # read csv
    _df, _usecols = _read_csv(csv_file=csv_file,
                              _usecols=_usecols,
                              csv_engine=csv_engine,
                              parquet_file=parquet_file,
                              sample=sample,
                              verbose=verbose)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _Pearson_screening_df_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            # returned Pearson correlation is a symmetric matrix
            _a -= _np.mean(_a)
            _a /= _np.std(_a)
            _b -= _np.mean(_b)
            _b /= _np.std(_b)
            #             return _np.corrcoef(_a, _b)[0, 1]
            return _a @ _b / len(_a)

        _pearson_slice = _np.array(list(map(_map_foo, _slice)))
        return _pearson_slice

    # multiprocessing starts here
    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome

    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        Pearson_df = pl.map(_Pearson_screening_df_slice, _iter)
    Pearson_df = _np.hstack(Pearson_df)

    del _df

    return Pearson_df


def clump_continuous_csv_parallel(
        csv_file="_",
        _usecols=[],
        a_N=300,
        b_N=300,
        kernel="epa",
        bw="silverman",
        bw_multiplier=1.,
        norm=2,
        clumping_threshold=Pearson_to_MI_Gaussian(.6),
        num_vars_exam=_np.infty,
        core_num="NOT DECLARED",
        multp=10,
        csv_engine="c",
        parquet_file="_",
        sample=256000,
        verbose=1,
        share_memory=True,
        **kwarg):
    """
    Perform clumping based on mutual information thresholding
    The clumping process starts from the left to right, preserve input variables under the clumping threshold
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.    """
    # initialization
    _, keep_cols = _read_csv(csv_file=csv_file,
                             _usecols=_usecols,
                             csv_engine="dask",
                             parquet_file=parquet_file,
                             sample=sample,
                             verbose=verbose)

    del _

    if num_vars_exam == _np.infty:
        num_vars_exam = len(keep_cols) - 1
    _iter = _np.arange(num_vars_exam)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    for current_var_ind in _iter:  # note that here _iter and keep_cols don't need to agree, by the break command comes later
        if current_var_ind + 1 <= len(keep_cols):
            _MI = continuous_screening_csv_parallel(
                csv_file=csv_file,
                _usecols=keep_cols[current_var_ind:],
                kernel=kernel,
                bw=bw,
                bw_multiplier=bw_multiplier,
                norm=norm,
                core_num=core_num,
                multp=multp,
                csv_engine=csv_engine,
                parquet_file=parquet_file,
                sample=sample,
                verbose=0,
                share_memory=share_memory,
                **kwarg)
            # current_var_ind + 1 since the current variable will be included anyway
            keep_cols = _np.hstack(
                (keep_cols[:current_var_ind + 1],
                 keep_cols[current_var_ind + 1:][_MI <= clumping_threshold]))
        else:
            break
    return current_var_ind, keep_cols


def binary_screening_dataframe(dataframe="_",
                               _usecols=[],
                               N=500,
                               kernel="epa",
                               bw="silverman",
                               bw_multiplier=1.,
                               csv_engine="c",
                               verbose=1,
                               **kwarg):
    """
    Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    The outcome should be binary and the covariates be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    """
    _df = dataframe
    if _np.array(_usecols).size == 0:
        _usecols = _np.array(_df.columns.to_list()[1:])
    else:
        _usecols = _np.array(_usecols)

    def _map_foo(j):
        __ = [
            _usecols[0], _usecols[j + 1]
        ]  # here using _usecol[j + 1] because the left first column is the outcome
        _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
        return MI_binary_continuous(a=_a,
                                    b=_b,
                                    N=N,
                                    kernel=kernel,
                                    bw=bw,
                                    bw_multiplier=bw_multiplier,
                                    **kwarg)

    _iter = _np.arange(len(_usecols) - 1)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    MI_df = _np.array(list(map(_map_foo, _iter)))

    return MI_df


def continuous_screening_dataframe(dataframe="_",
                                   _usecols=[],
                                   a_N=300,
                                   b_N=300,
                                   kernel="epa",
                                   bw="silverman",
                                   bw_multiplier=1.,
                                   norm=2,
                                   csv_engine="c",
                                   verbose=1,
                                   **kwarg):
    """
    Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    Both the outcome and the covariates should be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    """
    _df = dataframe
    if _np.array(_usecols).size == 0:
        _usecols = _np.array(_df.columns.to_list()[1:])
    else:
        _usecols = _np.array(_usecols)

    def _map_foo(j):
        __ = [
            _usecols[0], _usecols[j + 1]
        ]  # here using _usecol[j + 1] because the left first column is the outcome
        _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
        return MI_continuous_continuous(a=_a,
                                        b=_b,
                                        a_N=a_N,
                                        b_N=b_N,
                                        kernel=kernel,
                                        bw=bw,
                                        bw_multiplier=bw_multiplier,
                                        norm=norm,
                                        **kwarg)

    _iter = _np.arange(len(_usecols) - 1)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    MI_df = _np.array(list(map(_map_foo, _iter)))

    return MI_df


def binary_screening_dataframe_parallel(dataframe="_",
                                        _usecols=[],
                                        N=500,
                                        kernel="epa",
                                        bw="silverman",
                                        bw_multiplier=1.,
                                        core_num="NOT DECLARED",
                                        multp=10,
                                        csv_engine="c",
                                        verbose=1,
                                        share_memory=True,
                                        **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    The outcome should be binary and the covariates be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    _df = dataframe
    if _np.array(_usecols).size == 0:
        _usecols = _np.array(_df.columns.to_list()[1:])
    else:
        _usecols = _np.array(_usecols)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _binary_screening_csv_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return MI_binary_continuous(a=_a,
                                        b=_b,
                                        N=N,
                                        kernel=kernel,
                                        bw=bw,
                                        bw_multiplier=bw_multiplier,
                                        **kwarg)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here

    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome
    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_binary_screening_csv_slice, _iter)
    MI_df = _np.hstack(MI_df)

    return MI_df


def continuous_screening_dataframe_parallel(dataframe="_",
                                            _usecols=[],
                                            a_N=300,
                                            b_N=300,
                                            kernel="epa",
                                            bw="silverman",
                                            bw_multiplier=1.,
                                            norm=2,
                                            core_num="NOT DECLARED",
                                            multp=10,
                                            csv_engine="c",
                                            verbose=1,
                                            share_memory=True,
                                            **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    Both the outcome and the covariates should be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    _df = dataframe
    if _np.array(_usecols).size == 0:
        _usecols = _np.array(_df.columns.to_list()[1:])
    else:
        _usecols = _np.array(_usecols)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _continuous_screening_csv_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return MI_continuous_continuous(a=_a,
                                            b=_b,
                                            a_N=a_N,
                                            b_N=b_N,
                                            kernel=kernel,
                                            bw=bw,
                                            bw_multiplier=bw_multiplier,
                                            norm=norm,
                                            **kwarg)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome

    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_continuous_screening_csv_slice, _iter)
    MI_df = _np.hstack(MI_df)

    return MI_df


def binning_binary_screening_dataframe_parallel(dataframe="_",
                                                _usecols=[],
                                                core_num="NOT DECLARED",
                                                multp=10,
                                                csv_engine="c",
                                                verbose=1,
                                                share_memory=True,
                                                **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    The outcome should be binary and the covariates be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    _df = dataframe
    if _np.array(_usecols).size == 0:
        _usecols = _np.array(_df.columns.to_list()[1:])
    else:
        _usecols = _np.array(_usecols)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _binary_screening_csv_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return binning_MI_discrete_cython(a=_a, b=_b)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here

    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome
    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_binary_screening_csv_slice, _iter)
    MI_df = _np.hstack(MI_df)

    return MI_df


def binning_continuous_screening_dataframe_parallel(dataframe="_",
                                                    _usecols=[],
                                                    core_num="NOT DECLARED",
                                                    multp=10,
                                                    csv_engine="c",
                                                    verbose=1,
                                                    share_memory=True,
                                                    **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    Both the outcome and the covariates should be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    _df = dataframe
    if _np.array(_usecols).size == 0:
        _usecols = _np.array(_df.columns.to_list()[1:])
    else:
        _usecols = _np.array(_usecols)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _continuous_screening_csv_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return binning_MI_cython(a=_a, b=_b)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome

    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_continuous_screening_csv_slice, _iter)
    MI_df = _np.hstack(MI_df)

    return MI_df


def binary_skMI_screening_dataframe_parallel(dataframe="_",
                                             _usecols=[],
                                             n_neighbors=3,
                                             core_num="NOT DECLARED",
                                             multp=10,
                                             csv_engine="c",
                                             verbose=1,
                                             share_memory=True,
                                             **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    Both the outcome and the covariates should be binary. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    _df = dataframe
    if _np.array(_usecols).size == 0:
        _usecols = _np.array(_df.columns.to_list()[1:])
    else:
        _usecols = _np.array(_usecols)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _binary_skMI_df_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return _mutual_info_classif(y=_a.reshape(-1, 1),
                                        X=_b.reshape(-1, 1),
                                        n_neighbors=n_neighbors,
                                        discrete_features=False,
                                        **kwarg)[0]

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome

    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_binary_skMI_df_slice, _iter)
    MI_df = _np.hstack(MI_df)

    return MI_df


def continuous_skMI_screening_dataframe_parallel(dataframe="_",
                                                 _usecols=[],
                                                 n_neighbors=3,
                                                 core_num="NOT DECLARED",
                                                 multp=10,
                                                 csv_engine="c",
                                                 verbose=1,
                                                 share_memory=True,
                                                 **kwarg):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the mutual information between outcome and covariates.
    Both the outcome and the covariates should be continuous. 
    If _usecols is given, the returned mutual information will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.
    """
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    _df = dataframe
    if _np.array(_usecols).size == 0:
        _usecols = _np.array(_df.columns.to_list()[1:])
    else:
        _usecols = _np.array(_usecols)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _continuous_skMI_df_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            return _mutual_info_regression(y=_a.reshape(-1, 1),
                                           X=_b.reshape(-1, 1),
                                           n_neighbors=n_neighbors,
                                           discrete_features=False,
                                           **kwarg)[0]

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome

    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_df = pl.map(_continuous_skMI_df_slice, _iter)
    MI_df = _np.hstack(MI_df)

    return MI_df


def Pearson_screening_dataframe_parallel(dataframe="_",
                                         _usecols=[],
                                         core_num="NOT DECLARED",
                                         multp=10,
                                         csv_engine="c",
                                         verbose=1,
                                         share_memory=True):
    """
    (Multiprocessing version) Take a (potentionally large) csv file to calculate the Pearson's correlation between outcome and covariates.
    If _usecols is given, the returned Pearson correlation will match _usecols. 
    By default, the left first covariate should be the outcome -- use _usecols to adjust if not the case.
    This function accounts for missing data better than the Pearson's correlation matrix function provided by numpy.
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.    
    """
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    _df = dataframe
    if _np.array(_usecols).size == 0:
        _usecols = _np.array(_df.columns.to_list()[1:])
    else:
        _usecols = _np.array(_usecols)

    # share_memory for multiprocess
    if share_memory == True:
        # the origingal dataframe is df, store the columns/dtypes pairs
        df_dtypes_dict = dict(list(zip(_df.columns, _df.dtypes)))
        # declare a shared Array with data from df
        mparr = _mp.Array(_ctypes.c_double, _df.values.reshape(-1))
        # create a new df based on the shared array
        _df = _pd.DataFrame(_np.frombuffer(mparr.get_obj()).reshape(_df.shape),
                            columns=_df.columns).astype(df_dtypes_dict)

    def _Pearson_screening_df_slice(_slice):

        def _map_foo(j):
            __ = [
                _usecols[0], _usecols[j]
            ]  # here using _usecol[j] because only input variables indices were splitted
            _a, _b = _read_two_columns(_df=_df, __=__, csv_engine=csv_engine)
            # returned Pearson correlation is a symmetric matrix
            _a -= _np.mean(_a)
            _a /= _np.std(_a)
            _b -= _np.mean(_b)
            _b /= _np.std(_b)
            #             return _np.corrcoef(_a, _b)[0, 1]
            return _a @ _b / len(_a)

        _pearson_slice = _np.array(list(map(_map_foo, _slice)))
        return _pearson_slice

    # multiprocessing starts here
    ind = _np.arange(
        1, len(_usecols)
    )  # starting from 1 because the first left column should be the outcome

    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        Pearson_df = pl.map(_Pearson_screening_df_slice, _iter)
    Pearson_df = _np.hstack(Pearson_df)

    return Pearson_df


def clump_continuous_dataframe_parallel(
        dataframe="_",
        _usecols=[],
        a_N=300,
        b_N=300,
        kernel="epa",
        bw="silverman",
        bw_multiplier=1.,
        norm=2,
        clumping_threshold=Pearson_to_MI_Gaussian(.6),
        num_vars_exam=_np.infty,
        core_num="NOT DECLARED",
        multp=10,
        csv_engine="c",
        verbose=1,
        share_memory=True,
        **kwarg):
    """
    Perform clumping based on mutual information thresholding
    The clumping process starts from the left to right, preserve input variables under the clumping threshold
    share_memory is to indicate whether to share the dataframe in memory to 
    multiple processes -- if set to False, each process will copy the entire dataframe respectively. However, 
    to read very large dataframe using dask, this option should usually be turned off.    """
    # initialization
    _df = dataframe
    if _np.array(_usecols).size == 0:
        _usecols = _np.array(_df.columns.to_list()[1:])
    else:
        _usecols = _np.array(_usecols)

    if num_vars_exam == _np.infty:
        num_vars_exam = len(keep_cols) - 1
    _iter = _np.arange(num_vars_exam)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    for current_var_ind in _iter:  # note that here _iter and keep_cols don't need to agree, by the break command comes later
        if current_var_ind + 1 <= len(keep_cols):
            _MI = continuous_screening_dataframe_parallel(
                dataframe=dataframe,
                _usecols=keep_cols[current_var_ind:],
                kernel=kernel,
                bw=bw,
                bw_multiplier=bw_multiplier,
                norm=norm,
                core_num=core_num,
                multp=multp,
                csv_engine=csv_engine,
                parquet_file=parquet_file,
                sample=sample,
                verbose=0,
                share_memory=share_memory,
                **kwarg)
            # current_var_ind + 1 since the current variable will be included anyway
            keep_cols = _np.hstack(
                (keep_cols[:current_var_ind + 1],
                 keep_cols[current_var_ind + 1:][_MI <= clumping_threshold]))
        else:
            break
    return current_var_ind, keep_cols


def binary_screening_array(X,
                           y,
                           drop_na=True,
                           N=500,
                           kernel="epa",
                           bw="silverman",
                           bw_multiplier=1.,
                           verbose=1,
                           **kwarg):
    """
    Take a numpy file to calculate the mutual information between outcome and covariates.
    The outcome should be binary and the covariates be continuous. 
    If drop_na is set to be True, the NaN values will be dropped in a bivariate manner. 
    """

    def _map_foo(j):
        _a, _b = y.copy(), X[:, j].copy()
        if drop_na == True:
            _keep = _np.logical_not(
                _np.logical_or(_np.isnan(_a), _np.isnan(_b)))
            _a, _b = _a[_keep], _b[_keep]
        return MI_binary_continuous(a=_a,
                                    b=_b,
                                    N=N,
                                    kernel=kernel,
                                    bw=bw,
                                    bw_multiplier=bw_multiplier,
                                    **kwarg)

    _iter = _np.arange(X.shape[1])
    if verbose >= 1:
        _iter = _tqdm(_iter)
    MI_array = _np.array(list(map(_map_foo, _iter)))
    return MI_array


def continuous_screening_array(X,
                               y,
                               drop_na=True,
                               a_N=300,
                               b_N=300,
                               kernel="epa",
                               bw="silverman",
                               bw_multiplier=1.,
                               norm=2,
                               verbose=1,
                               **kwarg):
    """
    Take a numpy file to calculate the mutual information between outcome and covariates.
    The outcome should be continuous and the covariates be continuous. 
    If drop_na is set to be True, the NaN values will be dropped in a bivariate manner. 
    """

    def _map_foo(j):
        _a, _b = y.copy(), X[:, j].copy()
        if drop_na == True:
            _keep = _np.logical_not(
                _np.logical_or(_np.isnan(_a), _np.isnan(_b)))
            _a, _b = _a[_keep], _b[_keep]
        return MI_continuous_continuous(a=_a,
                                        b=_b,
                                        a_N=a_N,
                                        b_N=b_N,
                                        kernel=kernel,
                                        bw=bw,
                                        bw_multiplier=bw_multiplier,
                                        norm=norm,
                                        **kwarg)

    _iter = _np.arange(X.shape[1])
    if verbose >= 1:
        _iter = _tqdm(_iter)
    MI_array = _np.array(list(map(_map_foo, _iter)))
    return MI_array


def binary_screening_array_parallel(X,
                                    y,
                                    drop_na=True,
                                    N=500,
                                    kernel="epa",
                                    bw="silverman",
                                    bw_multiplier=1.,
                                    core_num="NOT DECLARED",
                                    multp=10,
                                    verbose=1,
                                    **kwarg):
    """
    (Multiprocessing version) Calculate the mutual information between outcome and covariates.
    The outcome should be binary and the covariates be continuous. 
    If drop_na is set to be True, the NaN values will be dropped in a bivariate manner. 
    """
    # check some basic things
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    def _binary_screening_array_slice(_slice):

        def _map_foo(j):
            _a, _b = y.copy(), X[:, j].copy()
            if drop_na == True:
                _keep = _np.logical_not(
                    _np.logical_or(_np.isnan(_a), _np.isnan(_b)))
                _a, _b = _a[_keep], _b[_keep]
            return MI_binary_continuous(a=_a,
                                        b=_b,
                                        N=N,
                                        kernel=kernel,
                                        bw=bw,
                                        bw_multiplier=bw_multiplier,
                                        **kwarg)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(
        X.shape[1]
    )  # starting from 1 because the first left column should be the outcome
    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_array = pl.map(_binary_screening_csv_slice, _iter)
    MI_array = _np.hstack(MI_array)
    return MI_array


def continuous_screening_array_parallel(X,
                                        y,
                                        drop_na=True,
                                        a_N=300,
                                        b_N=300,
                                        kernel="epa",
                                        bw="silverman",
                                        bw_multiplier=1.,
                                        norm=2,
                                        core_num="NOT DECLARED",
                                        multp=10,
                                        verbose=1,
                                        **kwarg):
    """
    (Multiprocessing version) Calculate the mutual information between outcome and covariates.
    The outcome should be continuous and the covariates be continuous. 
    If drop_na is set to be True, the NaN values will be dropped in a bivariate manner. 
    """
    # check some basic things
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    def _continuous_screening_array_slice(_slice):

        def _map_foo(j):
            _a, _b = y.copy(), X[:, j].copy()
            if drop_na == True:
                _keep = _np.logical_not(
                    _np.logical_or(_np.isnan(_a), _np.isnan(_b)))
                _a, _b = _a[_keep], _b[_keep]
            return MI_continuous_continuous(a=_a,
                                            b=_b,
                                            a_N=a_N,
                                            b_N=b_N,
                                            kernel=kernel,
                                            bw=bw,
                                            bw_multiplier=bw_multiplier,
                                            norm=norm,
                                            **kwarg)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(X.shape[1])
    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_array = pl.map(_continuous_screening_array_slice, _iter)
    MI_array = _np.hstack(MI_array)
    return MI_array


def binary_skMI_array_parallel(X,
                               y,
                               drop_na=True,
                               n_neighbors=3,
                               core_num="NOT DECLARED",
                               multp=10,
                               verbose=1,
                               **kwarg):
    """
    (Multiprocessing version) Calculate the mutual information using sklearn implementation between outcome and covariates.
    The outcome should be binary and the covariates be binary. 
    If drop_na is set to be True, the NaN values will be dropped in a bivariate manner. 
    """
    # check some basic things
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    def _binary_skMI_array_slice(_slice):

        def _map_foo(j):
            _a, _b = y.copy(), X[:, j].copy()
            if drop_na == True:
                _keep = _np.logical_not(
                    _np.logical_or(_np.isnan(_a), _np.isnan(_b)))
                _a, _b = _a[_keep], _b[_keep]
            return _mutual_info_classif(y=_a.reshape(-1, 1),
                                        X=_b.reshape(-1, 1),
                                        n_neighbors=n_neighbors,
                                        discrete_features=False,
                                        **kwarg)[0]

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(X.shape[1])
    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_array = pl.map(_binary_skMI_array_slice, _iter)
    MI_array = _np.hstack(MI_array)
    return MI_array


def continuous_skMI_array_parallel(X,
                                   y,
                                   drop_na=True,
                                   n_neighbors=3,
                                   core_num="NOT DECLARED",
                                   multp=10,
                                   verbose=1,
                                   **kwarg):
    """
    (Multiprocessing version) Calculate the mutual information using sklearn implementation between outcome and covariates.
    The outcome should be binary and the covariates be continuous. 
    If drop_na is set to be True, the NaN values will be dropped in a bivariate manner. 
    """
    # check some basic things
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    def _continuous_skMI_array_slice(_slice):

        def _map_foo(j):
            _a, _b = y.copy(), X[:, j].copy()
            if drop_na == True:
                _keep = _np.logical_not(
                    _np.logical_or(_np.isnan(_a), _np.isnan(_b)))
                _a, _b = _a[_keep], _b[_keep]
            return _mutual_info_regression(y=_a.reshape(-1, 1),
                                           X=_b.reshape(-1, 1),
                                           n_neighbors=n_neighbors,
                                           discrete_features=False,
                                           **kwarg)[0]

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(X.shape[1])
    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_array = pl.map(_continuous_skMI_array_slice, _iter)
    MI_array = _np.hstack(MI_array)
    return MI_array


def continuous_Pearson_array_parallel(X,
                                      y,
                                      drop_na=True,
                                      n_neighbors=3,
                                      core_num="NOT DECLARED",
                                      multp=10,
                                      verbose=1):
    """
    (Multiprocessing version) Calculate the mutual information using sklearn implementation between outcome and covariates.
    The outcome should be binary and the covariates be continuous. 
    If drop_na is set to be True, the NaN values will be dropped in a bivariate manner. 
    """
    # check some basic things
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    def _continuous_Pearson_array_slice(_slice):

        def _map_foo(j):
            _a, _b = y.copy(), X[:, j].copy()
            if drop_na == True:
                _keep = _np.logical_not(
                    _np.logical_or(_np.isnan(_a), _np.isnan(_b)))
                _a, _b = _a[_keep], _b[_keep]
            _a -= _np.mean(_a)
            _a /= _np.std(_a)
            _b -= _np.mean(_b)
            _b /= _np.std(_b)
            return _a @ _b / len(_a)

        _MI_slice = _np.array(list(map(_map_foo, _slice)))
        return _MI_slice

    # multiprocessing starts here
    ind = _np.arange(X.shape[1])
    _iter = _np.array_split(ind, core_num * multp)
    if verbose >= 1:
        _iter = _tqdm(_iter)
    with _mp.Pool(core_num) as pl:
        MI_array = pl.map(_continuous_Pearson_array_slice, _iter)
    MI_array = _np.hstack(MI_array)
    return MI_array


##################################################################
###################### LM AG numba version  ######################
##################################################################
@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _update_smooth_grad_convex_LM(N, X, beta_md, y):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    return 1 / N * X.T @ (X @ beta_md - y)


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _update_smooth_grad_SCAD_LM(N, X, beta_md, y, _lambda, a):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _update_smooth_grad_convex_LM(N=N, X=X, beta_md=beta_md,
                                         y=y) + SCAD_concave_grad(
                                             x=beta_md, lambda_=_lambda, a=a)


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _update_smooth_grad_MCP_LM(N, X, beta_md, y, _lambda, gamma):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _update_smooth_grad_convex_LM(
        N=N, X=X, beta_md=beta_md, y=y) + MCP_concave_grad(
            x=beta_md, lambda_=_lambda, gamma=gamma)


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _eval_obj_SCAD_LM(N, X, beta_md, y, _lambda, a, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X @ x_temp
    return (error.T @ error) / (2. * N) + _np.sum(
        SCAD(x_temp, lambda_=_lambda, a=a))


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _eval_obj_MCP_LM(N, X, beta_md, y, _lambda, gamma, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X @ x_temp
    return (error.T @ error) / (2 * N) + _np.sum(
        SCAD(x_temp, lambda_=_lambda, gamma=gamma))


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def lambda_max_LM(X, y):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    #     X_temp = X.copy()
    #     X_temp = X_temp[:,1:]
    #     X_temp -= _np.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= _np.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= _np.mean(y)
    #     y_temp /= _np.std(y)
    grad_at_0 = y @ X[:, 1:] / len(y)
    lambda_max = _np.linalg.norm(grad_at_0, ord=_np.infty)
    return lambda_max


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _strong_rule_seq_LM(X, y, beta_old, lambda_new, lambda_old):
    """
    Use sequential strong to determine which betas to be nullified next.
    """
    #     X_temp = X.copy()
    #     X_temp -= _np.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= _np.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= _np.mean(y)
    #     y_temp /= _np.std(y)
    grad = _np.abs((y - X[:, 1:] @ beta_old[1:]) @ X[:, 1:] / (2 * len(y)))
    eliminated = (grad < 2 * lambda_new - lambda_old
                  )  # True means the value gets eliminated
    eliminated = _np.hstack(
        (_np.array([False]),
         eliminated))  # because intercept coefficient is not penalized
    return eliminated


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def UAG_LM_SCAD_MCP(design_matrix,
                    outcome,
                    beta_0=_np.ones(1),
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
    if _np.all(beta_0 == _np.ones(1)):
        center_X = (X - 1 / N * _np.sum(X, 0).reshape(1, -1))
        cov = (y - _np.mean(y)) @ center_X
        X_var = _np.sum(X**2, 0)
        beta = cov / X_var
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        if _np.any(
                X[:, 0] != X[0, 0]
        ):  # check if design matrix has included a column for intercept or not
            intercept_design = _np.ones(N).reshape(-1, 1)
            X = _np.hstack((intercept_design, X))
            beta = _np.hstack((_np.array([0.]), beta))
    # passing other parameters
    p = X.shape[1]  # so here p includes the intercept design matrix column
    smooth_grad = _np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    #     L_convex = 1/N*_np.max(_np.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1 / N * (_np.linalg.eigvalsh(X @ X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        #         L = _np.max(_np.array([L_convex, 1./(a-1)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (a - 1)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restartings
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _update_smooth_grad_SCAD_LM(N=N,
                                                      X=X,
                                                      beta_md=beta_md,
                                                      y=y,
                                                      _lambda=_lambda,
                                                      a=a)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    else:
        #         L = _np.max(_np.array([L_convex, 1./(gamma)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (gamma)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _update_smooth_grad_MCP_LM(N=N,
                                                     X=X,
                                                     beta_md=beta_md,
                                                     y=y,
                                                     _lambda=_lambda,
                                                     gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    return k, beta_md


# def vanilla_proximal(self):
#     '''
#     Carry out optimization using vanilla gradient descent.
#     '''
#     if self.penalty == "SCAD":
#         L = max([self.L_convex, 1/(self.a-1)])
#         self.vanilla_stepsize = 1/L
#         self._eval_obj_SCAD_LM(self.beta_md, self.obj_value)
#         self._eval_obj_SCAD_LM(self.beta, self.obj_value_ORIGINAL)
#         self._eval_obj_SCAD_LM(self.beta_ag, self.obj_value_AG)
#         self.old_beta = self.beta_md - 10.
#         while not self.converged:
#             self.k += 1
#             if self.k <= self.maxit:
#                 self._update_smooth_grad_SCAD_LM()
#                 self.beta_md = self.soft_thresholding(self.beta_md - self.vanilla_stepsize*self.smooth_grad, self.vanilla_stepsize*self._lambda)
#                 self.converged = _np.all(_np.max(_np.abs(self.beta_md - self.old_beta)) < self.tol).item()
#                 self.old_beta = self.beta_md.copy()
#                 self._eval_obj_SCAD_LM(self.beta_md, self.obj_value)
#                 self._eval_obj_SCAD_LM(self.beta, self.obj_value_ORIGINAL)
#                 self._eval_obj_SCAD_LM(self.beta_ag, self.obj_value_AG)
#             else:
#                 break
#     else:
#         L = max([self.L_convex, 1/self.gamma])
#         self.vanilla_stepsize = 1/L
#         self._eval_obj_MCP_LM(self.beta_md, self.obj_value)
#         self._eval_obj_MCP_LM(self.beta, self.obj_value_ORIGINAL)
#         self._eval_obj_MCP_LM(self.beta_ag, self.obj_value_AG)
#         self.old_beta = self.beta_md - 10.
#         while not self.converged:
#             self.k += 1
#             if self.k <= self.maxit:
#                 self._update_smooth_grad_MCP_LM()
#                 self.beta_md = self.soft_thresholding(self.beta_md - self.vanilla_stepsize*self.smooth_grad, self.vanilla_stepsize*self._lambda)
#                 self.converged = _np.all(_np.max(_np.abs(self.beta_md - self.old_beta)) < self.tol).item()
#                 self.old_beta = self.beta_md.copy()
#                 self._eval_obj_MCP_LM(self.beta_md, self.obj_value)
#                 self._eval_obj_MCP_LM(self.beta, self.obj_value_ORIGINAL)
#                 self._eval_obj_MCP_LM(self.beta_ag, self.obj_value_AG)
#             else:
#                 break
#     return self.report_results()


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def solution_path_LM(design_matrix,
                     outcome,
                     lambda_,
                     beta_0=_np.ones(1),
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
    N = design_matrix.shape[0]
    if add_intercept_column == True:
        if _np.any(
                design_matrix[:, 0] != design_matrix[0, 0]
        ):  # check if design matrix has included a column for intercept or not
            intercept_design = _np.ones(N).reshape(-1, 1)
            _design_matrix = design_matrix.copy()
            _design_matrix = _np.hstack((intercept_design, _design_matrix))
        else:
            _design_matrix = design_matrix
    else:
        _design_matrix = design_matrix
    beta_mat = _np.zeros((len(lambda_) + 1, _design_matrix.shape[1]))
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


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _UAG_LM_SCAD_MCP_strongrule(design_matrix,
                                outcome,
                                beta_0=_np.ones(1),
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
    if _np.all(beta_0 == _np.ones(1)):
        center_X = (X - 1 / N * _np.sum(X, 0).reshape(1, -1))
        cov = (y - _np.mean(y)) @ center_X
        X_var = _np.sum(X**2, 0)
        beta = cov / X_var
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        if _np.any(
                X[:, 0] != X[0, 0]
        ):  # check if design matrix has included a column for intercept or not
            intercept_design = _np.ones(N).reshape(-1, 1)
            X = _np.hstack((intercept_design, X))
            beta = _np.hstack((_np.array([0.]), beta))
    if strongrule == True:
        _lambda_max = lambda_max_LM(X, y)
        p_original = X.shape[1]
        elim = _strong_rule_seq_LM(X,
                                   y,
                                   beta_old=_np.zeros(p_original),
                                   lambda_new=_lambda,
                                   lambda_old=_lambda_max)
        X = X[:, _np.logical_not(elim)]
        beta = beta[_np.logical_not(elim)]

    # passing other parameters
    p = X.shape[1]  # so here p includes the intercept design matrix column
    smooth_grad = _np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    #     L_convex = 1/N*_np.max(_np.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1 / N * (_np.linalg.eigvalsh(X @ X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        #         L = _np.max(_np.array([L_convex, 1./(a-1)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (a - 1)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2. / (
                    1. + (1. + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1. - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _update_smooth_grad_SCAD_LM(N=N,
                                                      X=X,
                                                      beta_md=beta_md,
                                                      y=y,
                                                      _lambda=_lambda,
                                                      a=a)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    else:
        #         L = _np.max(_np.array([L_convex, 1./(gamma)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (gamma)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1. + (1. + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1. - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _update_smooth_grad_MCP_LM(N=N,
                                                     X=X,
                                                     beta_md=beta_md,
                                                     y=y,
                                                     _lambda=_lambda,
                                                     gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)


#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
#     if strongrule == True:
#         _beta_output = _np.zeros((p_original))
# #         _ = _np.argwhere(_np.logical_not(elim)).flatten()
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


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def UAG_LM_SCAD_MCP_strongrule(design_matrix,
                               outcome,
                               beta_0=_np.ones(1),
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
    output_beta = _np.zeros(len(_elim))
    output_beta[_np.logical_not(_elim)] = _beta_md
    return _k, output_beta


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def solution_path_LM_strongrule(design_matrix,
                                outcome,
                                lambda_,
                                beta_0=_np.ones(1),
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
    N = design_matrix.shape[0]
    if add_intercept_column == True:
        if _np.any(
                design_matrix[:, 0] != design_matrix[0, 0]
        ):  # check if design matrix has included a column for intercept or not
            intercept_design = _np.ones(N).reshape(-1, 1)
            _design_matrix = _np.hstack((intercept_design, _design_matrix))
    beta_mat = _np.empty((len(lambda_) + 1, _design_matrix.shape[1]))
    beta_mat[0, :] = 0.
    _lambda_max = lambda_max_LM(_design_matrix, outcome)
    lambda_ = _np.hstack((_np.array([_lambda_max]), lambda_))
    elim = _np.array([False] * _design_matrix.shape[1])
    for j in range(len(lambda_) - 1):
        _elim = _strong_rule_seq_LM(X=_design_matrix,
                                    y=outcome,
                                    beta_old=beta_mat[j, :],
                                    lambda_new=lambda_[j + 1],
                                    lambda_old=lambda_[j])
        elim = _np.logical_and(elim, _elim)
        _beta_0 = beta_mat[j, :]
        _new_beta = _np.zeros(_design_matrix.shape[1])
        _new_beta[_np.logical_not(elim)] = UAG_LM_SCAD_MCP(
            design_matrix=_design_matrix[:, _np.logical_not(elim)],
            outcome=outcome,
            beta_0=_beta_0[_np.logical_not(elim)],
            tol=tol,
            maxit=maxit,
            _lambda=lambda_[j],
            penalty=penalty,
            a=a,
            gamma=gamma,
            add_intercept_column=False)[1]
        beta_mat[j + 1, :] = _new_beta
    return beta_mat[1:, :]


##################################################################
########### LM AG SNP version using bed-reader ###################
##################################################################
# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_convex_LM(N, SNP_ind, bed, beta_md, y,
                                      outcome_iid):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    p = len(list(bed.sid))
    gene_iid = _np.array(list(bed.iid))
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]
    # first calcualte _=X@beta_md-y
    _ = _np.zeros(N)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _ += _X * beta_md[j + 1]  # +1 because intercept
    _ += beta_md[0]  # add the intercept
    _ -= _y
    # then calculate _XTXbeta = X.T@X@beta_md = X.T@_
    _XTXbeta = _np.zeros(p)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _XTXbeta[j] = _X @ _
    _XTXbeta = _np.hstack((_np.array([_np.sum(_)]), _XTXbeta))
    del _
    return 1 / N * _XTXbeta


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_SCAD_LM(N, SNP_ind, bed, beta_md, y, outcome_iid,
                                    _lambda, a):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _SNP_update_smooth_grad_convex_LM(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid) + SCAD_concave_grad(
            x=beta_md, lambda_=_lambda, a=a)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_MCP_LM(N, SNP_ind, bed, beta_md, y, outcome_iid,
                                   _lambda, gamma):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _SNP_update_smooth_grad_convex_LM(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid) + MCP_concave_grad(
            x=beta_md, lambda_=_lambda, gamma=gamma)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_lambda_max_LM(bed, y, outcome_iid, N, SNP_ind):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    #     X_temp = X.copy()
    #     X_temp = X_temp[:,1:]
    #     X_temp -= _np.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= _np.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= _np.mean(y)
    #     y_temp /= _np.std(y)
    p = len(list(bed.sid))
    grad_at_0 = _SNP_update_smooth_grad_convex_LM(N=N,
                                                  SNP_ind=SNP_ind,
                                                  bed=bed,
                                                  beta_md=_np.zeros(p),
                                                  y=y,
                                                  outcome_iid=outcome_iid)
    return _np.linalg.norm(grad_at_0[1:], ord=_np.infty)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_UAG_LM_SCAD_MCP(bed_file,
                        bim_file,
                        fam_file,
                        outcome,
                        outcome_iid,
                        SNP_ind,
                        L_convex,
                        beta_0=_np.ones(1),
                        tol=1e-5,
                        maxit=500,
                        _lambda=.5,
                        penalty="SCAD",
                        a=3.7,
                        gamma=2.):
    '''
    Carry out the optimization for penalized LM for a fixed lambda.
    '''
    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    y = outcome
    p = bed.sid_count
    gene_iid = _np.array(list(bed.iid))
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    if _np.all(beta_0 == _np.ones(1)):
        _ = _np.zeros(p)
        _y = y[_np.intersect1d(outcome_iid,
                               gene_iid,
                               assume_unique=True,
                               return_indices=True)[1]]
        _y -= _np.mean(_y)
        for j in SNP_ind:
            _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
            _X = _X[gene_ind]  # get gene iid also in outcome iid
            _X -= _np.mean(_X)
            _[j] = _X @ _y / N
        beta = _  # _np.sign(_)
        beta = _np.hstack((_np.array([0]), beta))
    else:
        beta = beta_0
    # passing other parameters
    smooth_grad = _np.ones(p + 1)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        #         L = _np.max(_np.array([L_convex, 1./(a-1)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (a - 1)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_SCAD_LM(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                a=a)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    else:
        #         L = _np.max(_np.array([L_convex, 1./(gamma)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (gamma)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_MCP_LM(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    return k, beta_md


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_solution_path_LM(bed_file,
                         bim_file,
                         fam_file,
                         outcome,
                         outcome_iid,
                         lambda_,
                         L_convex,
                         SNP_ind,
                         beta_0=_np.ones(1),
                         tol=1e-5,
                         maxit=500,
                         penalty="SCAD",
                         a=3.7,
                         gamma=2.):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    p = bed.sid_count

    y = outcome
    gene_iid = _np.array(list(bed.iid))
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    _ = _np.zeros(p)
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    _y -= _np.mean(_y)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _X -= _np.mean(_X)
        _[j] = _X @ _y / N
    beta = _  # _np.sign(_)
    beta = _np.hstack((_np.array([0]), beta)).reshape(1, -1)

    beta_mat = _np.zeros((len(lambda_) + 1, p + 1))
    beta_mat = _np.repeat(beta, len(lambda_) + 1, axis=0)
    for j in range(len(lambda_)):
        beta_mat[j + 1, :] = SNP_UAG_LM_SCAD_MCP(bed_file=bed_file,
                                                 bim_file=bim_file,
                                                 fam_file=fam_file,
                                                 outcome=outcome,
                                                 SNP_ind=SNP_ind,
                                                 L_convex=L_convex,
                                                 beta_0=beta_mat[j, :],
                                                 tol=tol,
                                                 maxit=maxit,
                                                 _lambda=lambda_[j],
                                                 penalty=penalty,
                                                 outcome_iid=outcome_iid,
                                                 a=a,
                                                 gamma=gamma)[1]
    return beta_mat[1:, :]


##################################################################
######### LM AG SNP PCA version using bed-reader #################
##################################################################
# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_convex_LM_PCA(N, SNP_ind, bed, beta_md, y,
                                          outcome_iid, pca_p, pca):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    p = len(list(bed.sid))
    gene_iid = _np.array(list(bed.iid))
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]
    # first calcualte _=X@beta_md-y
    _ = _np.zeros(N)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _ += _X * beta_md[j + 1]  # +1 because intercept
    _ += beta_md[0]  # add the intercept
    _ += pca[gene_ind, :] @ beta_md[1:pca_p + 1]
    _ -= _y
    # then calculate _XTXbeta = X.T@X@beta_md = X.T@_
    _XTXbeta = _np.zeros(p)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _XTXbeta[j] = _X @ _
    _XTXbeta = _np.hstack(
        (_np.array([_np.sum(_)]), _ @ pca[gene_ind, :], _XTXbeta))
    del _
    return 1 / N * _XTXbeta


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_SCAD_LM_PCA(N, SNP_ind, bed, beta_md, y,
                                        outcome_iid, _lambda, a, pca_p, pca):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _SNP_update_smooth_grad_convex_LM_PCA(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid,
        pca_p=pca_p,
        pca=pca) + SCAD_concave_grad_PCA(
            x=beta_md, lambda_=_lambda, a=a, pca_p=pca_p)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_MCP_LM_PCA(N, SNP_ind, bed, beta_md, y,
                                       outcome_iid, _lambda, gamma, pca_p,
                                       pca):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _SNP_update_smooth_grad_convex_LM_PCA(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid,
        pca_p=pca_p,
        pca=pca) + MCP_concave_grad_PCA(
            x=beta_md, lambda_=_lambda, gamma=gamma, pca_p=pca_p)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_lambda_max_LM_PCA(bed, y, outcome_iid, N, SNP_ind):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    #     X_temp = X.copy()
    #     X_temp = X_temp[:,1:]
    #     X_temp -= _np.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= _np.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= _np.mean(y)
    #     y_temp /= _np.std(y)
    p = len(list(bed.sid))
    grad_at_0 = _SNP_update_smooth_grad_convex_LM_PCA(N=N,
                                                      SNP_ind=SNP_ind,
                                                      bed=bed,
                                                      beta_md=_np.zeros(p),
                                                      y=y,
                                                      outcome_iid=outcome_iid)
    return _np.linalg.norm(grad_at_0[1:], ord=_np.infty)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_UAG_LM_SCAD_MCP_PCA(bed_file,
                            bim_file,
                            fam_file,
                            outcome,
                            outcome_iid,
                            SNP_ind,
                            L_convex,
                            pca,
                            beta_0=_np.ones(1),
                            tol=1e-5,
                            maxit=500,
                            _lambda=.5,
                            penalty="SCAD",
                            a=3.7,
                            gamma=2.):
    '''
    Carry out the optimization for penalized LM for a fixed lambda.
    '''
    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    pca_p = pca.shape[1]
    y = outcome
    p = bed.sid_count
    gene_iid = _np.array(list(bed.iid))
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    if _np.all(beta_0 == _np.ones(1)):
        _ = _np.zeros(p)
        _y = y[_np.intersect1d(outcome_iid,
                               gene_iid,
                               assume_unique=True,
                               return_indices=True)[1]]
        _y -= _np.mean(_y)
        for j in SNP_ind:
            _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
            _X = _X[gene_ind]  # get gene iid also in outcome iid
            _X -= _np.mean(_X)
            _[j] = _X @ _y / N / _np.var(_X)
        beta = _  # _np.sign(_)
        _pca = _y @ pca[gene_ind, :] / N
        beta = _np.hstack((_np.array([_np.mean(_y)]), _pca, beta))
    else:
        beta = beta_0
    # passing other parameters
    smooth_grad = _np.ones(p + 1 + pca_p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        #         L = _np.max(_np.array([L_convex, 1./(a-1)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (a - 1)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_SCAD_LM_PCA(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                a=a,
                pca_p=pca_p,
                pca=pca)
            beta = soft_thresholding_PCA(x=beta - opt_lambda * smooth_grad,
                                         lambda_=opt_lambda * _lambda,
                                         pca_p=pca_p)
            beta_ag = soft_thresholding_PCA(x=beta_md - opt_beta * smooth_grad,
                                            lambda_=opt_beta * _lambda,
                                            pca_p=pca_p)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    else:
        #         L = _np.max(_np.array([L_convex, 1./(gamma)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (gamma)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_MCP_LM_PCA(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                gamma=gamma,
                pca_p=pca_p,
                pca=pca)
            beta = soft_thresholding_PCA(x=beta - opt_lambda * smooth_grad,
                                         lambda_=opt_lambda * _lambda,
                                         pca_p=pca_p)
            beta_ag = soft_thresholding_PCA(x=beta_md - opt_beta * smooth_grad,
                                            lambda_=opt_beta * _lambda,
                                            pca_p=pca_p)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    return k, beta_md


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_solution_path_LM_PCA(bed_file,
                             bim_file,
                             fam_file,
                             outcome,
                             outcome_iid,
                             lambda_,
                             L_convex,
                             SNP_ind,
                             pca,
                             beta_0=_np.ones(1),
                             tol=1e-5,
                             maxit=500,
                             penalty="SCAD",
                             a=3.7,
                             gamma=2.):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    pca_p = pca.shape[1]
    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    p = bed.sid_count

    y = outcome
    gene_iid = _np.array(list(bed.iid))
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    _ = _np.zeros(p)
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    _y -= _np.mean(_y)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _X -= _np.mean(_X)
        _[j] = _X @ _y / N / _np.var(_X)
    beta = _  # _np.sign(_)
    _pca = _y @ pca[gene_ind, :] / N
    beta = _np.hstack((_np.array([_np.mean(_y)]), _pca, beta)).reshape(1, -1)
    beta_mat = _np.repeat(beta, len(lambda_) + 1, axis=0)
    for j in range(len(lambda_)):
        beta_mat[j + 1, :] = SNP_UAG_LM_SCAD_MCP_PCA(bed_file=bed_file,
                                                     bim_file=bim_file,
                                                     fam_file=fam_file,
                                                     outcome=outcome,
                                                     SNP_ind=SNP_ind,
                                                     L_convex=L_convex,
                                                     pca=pca,
                                                     beta_0=beta_mat[j, :],
                                                     tol=tol,
                                                     maxit=maxit,
                                                     _lambda=lambda_[j],
                                                     penalty=penalty,
                                                     outcome_iid=outcome_iid,
                                                     a=a,
                                                     gamma=gamma)[1]
    return beta_mat[1:, :]


###################################################################################
########### LM AG SNP version using bed-reader, multiprocess ######################
###################################################################################
# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)


def _SNP_update_smooth_grad_convex_LM_parallel(N, SNP_ind, bed, beta_md, y,
                                               outcome_iid, core_num, multp):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    p = len(list(bed.sid))
    gene_iid = _np.array(list(bed.iid))
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]

    # first calcualte _=X@beta_md-y
    def __parallel_plus(_ind):
        import numpy as _np
        _X = bed.read(_np.s_[:, _ind],
                      dtype=_np.int8).flatten().reshape(-1, len(_ind))
        _X = _X[gene_ind, :]  # get gene iid also in outcome iid
        return _X @ beta_md[_ind + 1]
#         __ = _np.zeros(N)
#         for j in _ind:
#             _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
#             _X = _X[gene_ind]  # get gene iid also in outcome iid
#             __ += _X * beta_md[j + 1]
#         return __

# multiprocessing starts here

    _splited_array = _np.array_split(SNP_ind, core_num * multp)
    _splited_array = [
        __array for __array in _splited_array if __array.size != 0
    ]
    with _mp.Pool(core_num) as pl:
        _ = pl.map(__parallel_plus, _splited_array)
    _ = _np.array(_).sum(0)
    _ += beta_md[0]  # add the intercept
    _ -= _y

    # then calculate _XTXbeta = X.T@X@beta_md = X.T@_
    def __parallel_assign(_ind):
        import numpy as _np
        _X = bed.read(_np.s_[:, _ind],
                      dtype=_np.int8).flatten().reshape(-1, len(_ind))
        _X = _X[gene_ind, :]  # get gene iid also in outcome iid
        return _ @ _X
#         k = 0
#         __ = _np.zeros(len(_ind))
#         for j in _ind:
#             _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
#             _X = _X[gene_ind]  # get gene iid also in outcome iid
#             __[k] = _X @ _
#             k += 1
#         return __

# multiprocessing starts here

    _splited_array = _np.array_split(SNP_ind, core_num * multp)
    _splited_array = [
        __array for __array in _splited_array if __array.size != 0
    ]
    with _mp.Pool(core_num) as pl:
        _XTXbeta = pl.map(__parallel_assign, _splited_array)
    __XTXbeta = _np.hstack(_XTXbeta)
    _XTXbeta = _np.zeros(p + 1)
    _XTXbeta[SNP_ind + 1] = __XTXbeta
    _XTXbeta[0] = _np.sum(_)
    del _
    del __XTXbeta

    return 1 / N * _XTXbeta


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_SCAD_LM_parallel(N, SNP_ind, bed, beta_md, y,
                                             outcome_iid, _lambda, a, core_num,
                                             multp):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _SNP_update_smooth_grad_convex_LM_parallel(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid,
        core_num=core_num,
        multp=multp) + SCAD_concave_grad(x=beta_md, lambda_=_lambda, a=a)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_MCP_LM_parallel(N, SNP_ind, bed, beta_md, y,
                                            outcome_iid, _lambda, gamma,
                                            core_num, multp):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _SNP_update_smooth_grad_convex_LM_parallel(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid,
        core_num=core_num,
        multp=multp) + MCP_concave_grad(
            x=beta_md, lambda_=_lambda, gamma=gamma)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_lambda_max_LM_parallel(bed, y, outcome_iid, N, SNP_ind, core_num,
                                multp):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    #     X_temp = X.copy()
    #     X_temp = X_temp[:,1:]
    #     X_temp -= _np.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= _np.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= _np.mean(y)
    #     y_temp /= _np.std(y)
    grad_at_0 = _SNP_update_smooth_grad_convex_LM_parallel(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=_np.zeros(len(SNP_ind)),
        y=y,
        outcome_iid=outcome_iid,
        core_num=core_num,
        multp=multp)
    return _np.linalg.norm(grad_at_0[1:], ord=_np.infty)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_UAG_LM_SCAD_MCP_parallel(bed_file,
                                 bim_file,
                                 fam_file,
                                 outcome,
                                 outcome_iid,
                                 SNP_ind,
                                 L_convex,
                                 beta_0=_np.ones(1),
                                 tol=1e-5,
                                 maxit=500,
                                 _lambda=.5,
                                 penalty="SCAD",
                                 a=3.7,
                                 gamma=2.,
                                 core_num="NOT DECLARED",
                                 multp=1):
    '''
    Carry out the optimization for penalized LM for a fixed lambda.
    '''
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    y = outcome
    p = bed.sid_count
    gene_iid = _np.array(list(bed.iid))
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    if _np.all(beta_0 == _np.ones(1)):
        _y = y[_np.intersect1d(outcome_iid,
                               gene_iid,
                               assume_unique=True,
                               return_indices=True)[1]]
        _y -= _np.mean(_y)

        def __parallel_assign(_ind):
            import numpy as _np
            _X = bed.read(_np.s_[:, _ind],
                          dtype=_np.float64).flatten().reshape(-1, len(_ind))
            _X = _X[gene_ind, :]  # get gene iid also in outcome iid
            _X -= _np.mean(_X, 0).reshape(1, -1)
            return _y @ _X / N
#             k = 0
#             __ = _np.zeros(len(_ind))
#             for j in _ind:
#                 _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
#                 _X = _X[gene_ind]  # get gene iid also in outcome iid
#                 _X -= _np.mean(_X)
#                 __[k] = _X @ _y / N
#                 k += 1
#             return __

# multiprocessing starts here

        _splited_array = _np.array_split(SNP_ind, core_num * multp)
        _splited_array = [
            __array for __array in _splited_array if __array.size != 0
        ]
        with _mp.Pool(core_num) as pl:
            _XTy = pl.map(__parallel_assign, _splited_array)
        _XTy = _np.hstack(_XTy)
        beta = _np.zeros(p + 1)
        beta[SNP_ind + 1] = _  # _np.sign(_XTy)
    else:
        beta = beta_0
    # passing other parameters
    smooth_grad = _np.ones(p + 1)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        #         L = _np.max(_np.array([L_convex, 1./(a-1)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (a - 1)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_SCAD_LM_parallel(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                a=a,
                core_num=core_num,
                multp=multp)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    else:
        #         L = _np.max(_np.array([L_convex, 1./(gamma)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (gamma)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_MCP_LM_parallel(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                gamma=gamma,
                core_num=core_num,
                multp=multp)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    return k, beta_md


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_solution_path_LM_parallel(bed_file,
                                  bim_file,
                                  fam_file,
                                  outcome,
                                  outcome_iid,
                                  lambda_,
                                  L_convex,
                                  SNP_ind,
                                  beta_0=_np.ones(1),
                                  tol=1e-5,
                                  maxit=500,
                                  penalty="SCAD",
                                  a=3.7,
                                  gamma=2.,
                                  core_num="NOT DECLARED",
                                  multp=1):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    y = outcome
    p = bed.sid_count
    gene_iid = _np.array(list(bed.iid))
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    _y -= _np.mean(_y)

    def __parallel_assign(_ind):
        import numpy as _np
        _X = bed.read(_np.s_[:, _ind],
                      dtype=_np.float64).flatten().reshape(-1, len(_ind))
        _X = _X[gene_ind, :]  # get gene iid also in outcome iid
        _X -= _np.mean(_X, 0).reshape(1, -1)
        return _y @ _X / N

#         k = 0
#         __ = _np.zeros(len(_ind))
#         for j in _ind:
#             _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
#             _X = _X[gene_ind]  # get gene iid also in outcome iid
#             _X -= _np.mean(_X)
#             __[k] = _X @ _y / N
#             k += 1
#         return __

# multiprocessing starts here

    _splited_array = _np.array_split(SNP_ind, core_num * multp)
    _splited_array = [
        __array for __array in _splited_array if __array.size != 0
    ]
    with _mp.Pool(core_num) as pl:
        _XTy = pl.map(__parallel_assign, _splited_array)
    _XTy = _np.hstack(_XTy)
    beta = _np.zeros(p + 1)
    beta[SNP_ind + 1] = _XTy  # _np.sign(_XTy)
    beta = beta.reshape(1, -1)

    beta_mat = _np.zeros((len(lambda_) + 1, p + 1))
    beta_mat = _np.repeat(beta, len(lambda_) + 1, axis=0)
    for j in range(len(lambda_)):
        beta_mat[j + 1, :] = SNP_UAG_LM_SCAD_MCP_parallel(
            bed_file=bed_file,
            bim_file=bim_file,
            fam_file=fam_file,
            outcome=outcome,
            SNP_ind=SNP_ind,
            L_convex=L_convex,
            beta_0=beta_mat[j, :],
            tol=tol,
            maxit=maxit,
            _lambda=lambda_[j],
            penalty=penalty,
            outcome_iid=outcome_iid,
            a=a,
            gamma=gamma,
            core_num=core_num,
            multp=multp)[1]
    return beta_mat[1:, :]


##################################################################
################### logistic AG numba version  ###################
##################################################################
@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _update_smooth_grad_convex_logistic(N, X, beta_md, y):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    return (X.T @ (_np.tanh(X @ beta_md / 2.) / 2. - y + .5)) / (2. * N)


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _update_smooth_grad_SCAD_logistic(N, X, beta_md, y, _lambda, a):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _update_smooth_grad_convex_logistic(
        N=N, X=X, beta_md=beta_md, y=y) + SCAD_concave_grad(
            x=beta_md, lambda_=_lambda, a=a)


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _update_smooth_grad_MCP_logistic(N, X, beta_md, y, _lambda, gamma):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _update_smooth_grad_convex_logistic(
        N=N, X=X, beta_md=beta_md, y=y) + MCP_concave_grad(
            x=beta_md, lambda_=_lambda, gamma=gamma)


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _eval_obj_SCAD_logistic(N, X, beta_md, y, _lambda, a, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X @ x_temp
    return (error.T @ error) / (2. * N) + _np.sum(
        SCAD(x_temp, lambda_=_lambda, a=a))


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _eval_obj_MCP_logistic(N, X, beta_md, y, _lambda, gamma, x_temp):
    '''
    evaluate value of the objective function.
    '''
    error = y - X @ x_temp
    return (error.T @ error) / (2 * N) + _np.sum(
        SCAD(x_temp, lambda_=_lambda, gamma=gamma))


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def lambda_max_logistic(X, y):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    grad_at_0 = (y - _np.mean(y)) @ X_temp / (2 * len(y))
    lambda_max = _np.linalg.norm(grad_at_0[1:], ord=_np.infty)
    return lambda_max


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _strong_rule_seq_logistic(X, y, beta_old, lambda_new, lambda_old):
    """
    Use sequential strong to determine which betas to be nullified next.
    """
    grad = _np.abs(
        (y - _np.tanh(X @ beta_old / 2) / 2 - .5) @ X_temp / (2 * len(y)))
    eliminated = (grad < 2 * lambda_new - lambda_old
                  )  # True means the value gets eliminated
    eliminated = _np.hstack(
        (_np.array([False]),
         eliminated))  # because intercept coefficient is not penalized
    return eliminated


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def UAG_logistic_SCAD_MCP(design_matrix,
                          outcome,
                          beta_0=_np.ones(1),
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
    if _np.all(beta_0 == _np.ones(1)):
        center_X = (X - 1 / N * _np.sum(X, 0).reshape(1, -1))
        cov = (y - _np.mean(y)) @ center_X
        X_var = _np.sum(X**2, 0)
        beta = cov / X_var
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        if _np.any(
                X[:, 0] != X[0, 0]
        ):  # check if design matrix has included a column for intercept or not
            intercept_design = _np.ones(N).reshape(-1, 1)
            X = _np.hstack((intercept_design, X))
            beta = _np.hstack((_np.array([0.]), beta))
    # passing other parameters
    p = X.shape[1]  # so here p includes the intercept design matrix column
    smooth_grad = _np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    #     L_convex = 1/N*_np.max(_np.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1 / N * (_np.linalg.eigvalsh(X @ X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        #         L = _np.max(_np.array([L_convex, 1./(a-1)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (a - 1)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _update_smooth_grad_SCAD_logistic(N=N,
                                                            X=X,
                                                            beta_md=beta_md,
                                                            y=y,
                                                            _lambda=_lambda,
                                                            a=a)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    else:
        #         L = _np.max(_np.array([L_convex, 1./(gamma)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (gamma)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _update_smooth_grad_MCP_logistic(N=N,
                                                           X=X,
                                                           beta_md=beta_md,
                                                           y=y,
                                                           _lambda=_lambda,
                                                           gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    return k, beta_md


# def vanilla_proximal(self):
#     '''
#     Carry out optimization using vanilla gradient descent.
#     '''
#     if self.penalty == "SCAD":
#         L = max([self.L_convex, 1/(self.a-1)])
#         self.vanilla_stepsize = 1/L
#         self._eval_obj_SCAD_logistic(self.beta_md, self.obj_value)
#         self._eval_obj_SCAD_logistic(self.beta, self.obj_value_ORIGINAL)
#         self._eval_obj_SCAD_logistic(self.beta_ag, self.obj_value_AG)
#         self.old_beta = self.beta_md - 10.
#         while not self.converged:
#             self.k += 1
#             if self.k <= self.maxit:
#                 self._update_smooth_grad_SCAD_logistic()
#                 self.beta_md = self.soft_thresholding(self.beta_md - self.vanilla_stepsize*self.smooth_grad, self.vanilla_stepsize*self._lambda)
#                 self.converged = _np.all(_np.max(_np.abs(self.beta_md - self.old_beta)) < self.tol).item()
#                 self.old_beta = self.beta_md.copy()
#                 self._eval_obj_SCAD_logistic(self.beta_md, self.obj_value)
#                 self._eval_obj_SCAD_logistic(self.beta, self.obj_value_ORIGINAL)
#                 self._eval_obj_SCAD_logistic(self.beta_ag, self.obj_value_AG)
#             else:
#                 break
#     else:
#         L = max([self.L_convex, 1/self.gamma])
#         self.vanilla_stepsize = 1/L
#         self._eval_obj_MCP_logistic(self.beta_md, self.obj_value)
#         self._eval_obj_MCP_logistic(self.beta, self.obj_value_ORIGINAL)
#         self._eval_obj_MCP_logistic(self.beta_ag, self.obj_value_AG)
#         self.old_beta = self.beta_md - 10.
#         while not self.converged:
#             self.k += 1
#             if self.k <= self.maxit:
#                 self._update_smooth_grad_MCP_logistic()
#                 self.beta_md = self.soft_thresholding(self.beta_md - self.vanilla_stepsize*self.smooth_grad, self.vanilla_stepsize*self._lambda)
#                 self.converged = _np.all(_np.max(_np.abs(self.beta_md - self.old_beta)) < self.tol).item()
#                 self.old_beta = self.beta_md.copy()
#                 self._eval_obj_MCP_logistic(self.beta_md, self.obj_value)
#                 self._eval_obj_MCP_logistic(self.beta, self.obj_value_ORIGINAL)
#                 self._eval_obj_MCP_logistic(self.beta_ag, self.obj_value_AG)
#             else:
#                 break
#     return self.report_results()


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def solution_path_logistic(design_matrix,
                           outcome,
                           lambda_,
                           beta_0=_np.ones(1),
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
    N = design_matrix.shape[0]
    if add_intercept_column == True:
        if _np.any(
                design_matrix[:, 0] != design_matrix[0, 0]
        ):  # check if design matrix has included a column for intercept or not
            intercept_design = _np.ones(N).reshape(-1, 1)
            _design_matrix = design_matrix.copy()
            _design_matrix = _np.hstack((intercept_design, _design_matrix))
        else:
            _design_matrix = design_matrix
    else:
        _design_matrix = design_matrix
    beta_mat = _np.zeros((len(lambda_) + 1, _design_matrix.shape[1]))
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


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _UAG_logistic_SCAD_MCP_strongrule(design_matrix,
                                      outcome,
                                      beta_0=_np.ones(1),
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
    if _np.all(beta_0 == _np.ones(1)):
        center_X = (X - 1 / N * _np.sum(X, 0).reshape(1, -1))
        cov = (y - _np.mean(y)) @ center_X
        X_var = _np.sum(X**2, 0)
        beta = cov / X_var
    else:
        beta = beta_0
#     add design matrix column for the intercept, if it's not there already
    if add_intercept_column == True:
        if _np.any(
                X[:, 0] != X[0, 0]
        ):  # check if design matrix has included a column for intercept or not
            intercept_design = _np.ones(N).reshape(-1, 1)
            X = _np.hstack((intercept_design, X))
            beta = _np.hstack((_np.array([0.]), beta))
    if strongrule == True:
        _lambda_max = lambda_max_logistic(X, y)
        p_original = X.shape[1]
        elim = _strong_rule_seq_logistic(X,
                                         y,
                                         beta_old=_np.zeros(p_original),
                                         lambda_new=_lambda,
                                         lambda_old=_lambda_max)
        X = X[:, _np.logical_not(elim)]
        beta = beta[_np.logical_not(elim)]

    # passing other parameters
    p = X.shape[1]  # so here p includes the intercept design matrix column
    smooth_grad = _np.ones(p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    #     L_convex = 1/N*_np.max(_np.linalg.eigvalsh(X@X.T)[-1]).item()
    if L_convex == 1.1:
        L_convex = 1 / N * (_np.linalg.eigvalsh(X @ X.T)[-1])
    else:
        pass
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        #         L = _np.max(_np.array([L_convex, 1./(a-1)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (a - 1)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2. / (
                    1. + (1. + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1. - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _update_smooth_grad_SCAD_logistic(N=N,
                                                            X=X,
                                                            beta_md=beta_md,
                                                            y=y,
                                                            _lambda=_lambda,
                                                            a=a)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    else:
        #         L = _np.max(_np.array([L_convex, 1./(gamma)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (gamma)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1. + (1. + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1. - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _update_smooth_grad_MCP_logistic(N=N,
                                                           X=X,
                                                           beta_md=beta_md,
                                                           y=y,
                                                           _lambda=_lambda,
                                                           gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)


#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
#     if strongrule == True:
#         _beta_output = _np.zeros((p_original))
# #         _ = _np.argwhere(_np.logical_not(elim)).flatten()
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


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def UAG_logistic_SCAD_MCP_strongrule(design_matrix,
                                     outcome,
                                     beta_0=_np.ones(1),
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
    output_beta = _np.zeros(len(_elim))
    output_beta[_np.logical_not(_elim)] = _beta_md
    return _k, output_beta


@_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def solution_path_logistic_strongrule(design_matrix,
                                      outcome,
                                      lambda_,
                                      beta_0=_np.ones(1),
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
    N = design_matrix.shape[0]
    if add_intercept_column == True:
        if _np.any(
                design_matrix[:, 0] != design_matrix[0, 0]
        ):  # check if design matrix has included a column for intercept or not
            intercept_design = _np.ones(N).reshape(-1, 1)
            _design_matrix = _np.hstack((intercept_design, _design_matrix))
    beta_mat = _np.empty((len(lambda_) + 1, _design_matrix.shape[1]))
    beta_mat[0, :] = 0.
    _lambda_max = lambda_max_logistic(_design_matrix, outcome)
    lambda_ = _np.hstack((_np.array([_lambda_max]), lambda_))
    elim = _np.array([False] * _design_matrix.shape[1])
    for j in range(len(lambda_) - 1):
        _elim = _strong_rule_seq_logistic(X=_design_matrix,
                                          y=outcome,
                                          beta_old=beta_mat[j, :],
                                          lambda_new=lambda_[j + 1],
                                          lambda_old=lambda_[j])
        elim = _np.logical_and(elim, _elim)
        _beta_0 = beta_mat[j, :]
        _new_beta = _np.zeros(_design_matrix.shape[1])
        _new_beta[_np.logical_not(elim)] = UAG_logistic_SCAD_MCP(
            design_matrix=_design_matrix[:, _np.logical_not(elim)],
            outcome=outcome,
            beta_0=_beta_0[_np.logical_not(elim)],
            tol=tol,
            maxit=maxit,
            _lambda=lambda_[j],
            penalty=penalty,
            a=a,
            gamma=gamma,
            add_intercept_column=False)[1]
        beta_mat[j + 1, :] = _new_beta
    return beta_mat[1:, :]


############################################################################
############# logistic SNP version with bed-reader #########################
############################################################################
# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_convex_logistic(N, SNP_ind, bed, beta_md, y,
                                            outcome_iid):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    p = len(list(bed.sid))
    gene_iid = _np.array(list(bed.iid))
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]
    # first calcualte _=X@beta_md-_y
    _ = _np.zeros(N)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _ += _X * beta_md[j + 1]  # +1 because intercept
    _ += beta_md[0]  # add the intercept
    _ = _np.tanh(_ / 2.) / 2. - _y + .5
    # then calculate output
    _XTXbeta = _np.zeros(p)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _XTXbeta[j] = _X @ _
    _XTXbeta = _np.hstack((_np.array([_np.sum(_)]), _XTXbeta))
    del _
    return _XTXbeta / (2. * N)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_SCAD_logistic(N, SNP_ind, bed, beta_md, y,
                                          outcome_iid, _lambda, a):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _SNP_update_smooth_grad_convex_logistic(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid) + SCAD_concave_grad(
            x=beta_md, lambda_=_lambda, a=a)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_MCP_logistic(N, SNP_ind, bed, beta_md, y,
                                         outcome_iid, _lambda, gamma):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _SNP_update_smooth_grad_convex_logistic(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid) + MCP_concave_grad(
            x=beta_md, lambda_=_lambda, gamma=gamma)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_lambda_max_logistic(bed, y, outcome_iid, N, SNP_ind):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    #     X_temp = X.copy()
    #     X_temp = X_temp[:,1:]
    #     X_temp -= _np.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= _np.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= _np.mean(y)
    #     y_temp /= _np.std(y)
    p = len(list(bed.sid))
    grad_at_0 = _SNP_update_smooth_grad_convex_logistic(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=_np.zeros(p),
        y=y,
        outcome_iid=outcome_iid)
    return _np.linalg.norm(grad_at_0[1:], ord=_np.infty)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_UAG_logistic_SCAD_MCP(bed_file,
                              bim_file,
                              fam_file,
                              outcome,
                              outcome_iid,
                              SNP_ind,
                              L_convex,
                              beta_0=_np.ones(1),
                              tol=1e-5,
                              maxit=500,
                              _lambda=.5,
                              penalty="SCAD",
                              a=3.7,
                              gamma=2.):
    '''
    Carry out the optimization for penalized logistic for a fixed lambda.
    '''
    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    y = outcome
    p = bed.sid_count
    gene_iid = _np.array(list(bed.iid))
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    if _np.all(beta_0 == _np.ones(1)):
        _ = _np.zeros(p)
        _y = y[_np.intersect1d(outcome_iid,
                               gene_iid,
                               assume_unique=True,
                               return_indices=True)[1]]
        _y = _y.astype(dtype=_np.float64)
        _y -= _np.mean(_y)
        for j in SNP_ind:
            _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
            _X = _X[gene_ind]  # get gene iid also in outcome iid
            _X -= _np.mean(_X)
            _[j] = _X @ _y / N
        beta = _np.sign(_)
        beta = _np.hstack((_np.array([0]), beta))
    else:
        beta = beta_0
    # passing other parameters
    smooth_grad = _np.ones(p + 1)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        #         L = _np.max(_np.array([L_convex, 1./(a-1)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (a - 1)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_SCAD_logistic(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                a=a)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    else:
        #         L = _np.max(_np.array([L_convex, 1./(gamma)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (gamma)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_MCP_logistic(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                gamma=gamma)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    return k, beta_md


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_solution_path_logistic(bed_file,
                               bim_file,
                               fam_file,
                               outcome,
                               outcome_iid,
                               lambda_,
                               L_convex,
                               SNP_ind,
                               beta_0=_np.ones(1),
                               tol=1e-5,
                               maxit=500,
                               penalty="SCAD",
                               a=3.7,
                               gamma=2.):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    p = bed.sid_count

    y = outcome
    gene_iid = _np.array(list(bed.iid))
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    _ = _np.zeros(p)
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    _y = _y.astype(dtype=_np.float64)
    _y -= _np.mean(_y)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _X -= _np.mean(_X)
        _[j] = _X @ _y / N
    beta = _  # _np.sign(_)
    beta = _np.hstack((_np.array([0]), beta)).reshape(1, -1)

    beta_mat = _np.zeros((len(lambda_) + 1, p + 1))
    beta_mat = _np.repeat(beta, len(lambda_) + 1, axis=0)
    for j in range(len(lambda_)):
        beta_mat[j + 1, :] = SNP_UAG_logistic_SCAD_MCP(bed_file=bed_file,
                                                       bim_file=bim_file,
                                                       fam_file=fam_file,
                                                       outcome=outcome,
                                                       SNP_ind=SNP_ind,
                                                       L_convex=L_convex,
                                                       beta_0=beta_mat[j, :],
                                                       tol=tol,
                                                       maxit=maxit,
                                                       _lambda=lambda_[j],
                                                       penalty=penalty,
                                                       outcome_iid=outcome_iid,
                                                       a=a,
                                                       gamma=gamma)[1]
    return beta_mat[1:, :]


############################################################################
########### logistic SNP PCA version with bed-reader #######################
############################################################################
# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_convex_logistic_PCA(N, SNP_ind, bed, beta_md, y,
                                                outcome_iid, pca_p, pca):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    p = len(list(bed.sid))
    gene_iid = _np.array(list(bed.iid))
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]
    # first calcualte _=X@beta_md-y
    _ = _np.zeros(N)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _ += _X * beta_md[j + 1]  # +1 because intercept
    _ += beta_md[0]  # add the intercept
    _ += pca[gene_ind, :] @ beta_md[1:pca_p + 1]
    _ = _np.tanh(_ / 2.) / 2. - _y + .5
    # then calculate _XTXbeta = X.T@X@beta_md = X.T@_
    _XTXbeta = _np.zeros(p)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _XTXbeta[j] = _X @ _
    _XTXbeta = _np.hstack(
        (_np.array([_np.sum(_)]), _ @ pca[gene_ind, :], _XTXbeta))
    del _
    return 1 / N * _XTXbeta


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_SCAD_logistic_PCA(N, SNP_ind, bed, beta_md, y,
                                              outcome_iid, _lambda, a, pca_p,
                                              pca):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _SNP_update_smooth_grad_convex_logistic_PCA(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid,
        pca_p=pca_p,
        pca=pca) + SCAD_concave_grad_PCA(
            x=beta_md, lambda_=_lambda, a=a, pca_p=pca_p)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_MCP_logistic_PCA(N, SNP_ind, bed, beta_md, y,
                                             outcome_iid, _lambda, gamma,
                                             pca_p, pca):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _SNP_update_smooth_grad_convex_logistic_PCA(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid,
        pca_p=pca_p,
        pca=pca) + MCP_concave_grad_PCA(
            x=beta_md, lambda_=_lambda, gamma=gamma, pca_p=pca_p)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_lambda_max_logistic_PCA(bed, y, outcome_iid, N, SNP_ind):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    #     X_temp = X.copy()
    #     X_temp = X_temp[:,1:]
    #     X_temp -= _np.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= _np.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= _np.mean(y)
    #     y_temp /= _np.std(y)
    p = len(list(bed.sid))
    grad_at_0 = _SNP_update_smooth_grad_convex_logistic_PCA(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=_np.zeros(p),
        y=y,
        outcome_iid=outcome_iid)
    return _np.linalg.norm(grad_at_0[1:], ord=_np.infty)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_UAG_logistic_SCAD_MCP_PCA(bed_file,
                                  bim_file,
                                  fam_file,
                                  outcome,
                                  outcome_iid,
                                  SNP_ind,
                                  L_convex,
                                  pca,
                                  beta_0=_np.ones(1),
                                  tol=1e-5,
                                  maxit=500,
                                  _lambda=.5,
                                  penalty="SCAD",
                                  a=3.7,
                                  gamma=2.):
    '''
    Carry out the optimization for penalized logistic for a fixed lambda.
    '''
    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    pca_p = pca.shape[1]
    y = outcome
    p = bed.sid_count
    gene_iid = _np.array(list(bed.iid))
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    if _np.all(beta_0 == _np.ones(1)):
        _ = _np.zeros(p)
        _y = y[_np.intersect1d(outcome_iid,
                               gene_iid,
                               assume_unique=True,
                               return_indices=True)[1]]
        _y = _y.astype(dtype=_np.float64)
        _y -= _np.mean(_y)
        for j in SNP_ind:
            _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
            _X = _X[gene_ind]  # get gene iid also in outcome iid
            _X -= _np.mean(_X)
            _[j] = _X @ _y / N / _np.var(_X)
        beta = _  # _np.sign(_)
        _pca = _y @ pca[gene_ind, :] / N / _np.var(pca[gene_ind, :], 0)
        beta = _np.hstack((_np.array([0.]), _pca, beta))
#         beta = _np.sign(beta)
    else:
        beta = beta_0
    # passing other parameters
    smooth_grad = _np.ones(p + 1 + pca_p)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        #         L = _np.max(_np.array([L_convex, 1./(a-1)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (a - 1)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_SCAD_logistic_PCA(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                a=a,
                pca_p=pca_p,
                pca=pca)
            beta = soft_thresholding_PCA(x=beta - opt_lambda * smooth_grad,
                                         lambda_=opt_lambda * _lambda,
                                         pca_p=pca_p)
            beta_ag = soft_thresholding_PCA(x=beta_md - opt_beta * smooth_grad,
                                            lambda_=opt_beta * _lambda,
                                            pca_p=pca_p)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    else:
        #         L = _np.max(_np.array([L_convex, 1./(gamma)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (gamma)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_MCP_logistic_PCA(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                gamma=gamma,
                pca_p=pca_p,
                pca=pca)
            beta = soft_thresholding_PCA(x=beta - opt_lambda * smooth_grad,
                                         lambda_=opt_lambda * _lambda,
                                         pca_p=pca_p)
            beta_ag = soft_thresholding_PCA(x=beta_md - opt_beta * smooth_grad,
                                            lambda_=opt_beta * _lambda,
                                            pca_p=pca_p)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    return k, beta_md


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_solution_path_logistic_PCA(bed_file,
                                   bim_file,
                                   fam_file,
                                   outcome,
                                   outcome_iid,
                                   lambda_,
                                   L_convex,
                                   SNP_ind,
                                   pca,
                                   beta_0=_np.ones(1),
                                   tol=1e-5,
                                   maxit=500,
                                   penalty="SCAD",
                                   a=3.7,
                                   gamma=2.):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    pca_p = pca.shape[1]
    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    p = bed.sid_count

    y = outcome
    gene_iid = _np.array(list(bed.iid))
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    _ = _np.zeros(p)
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    _y = _y.astype(dtype=_np.float64)
    _y -= _np.mean(_y)
    for j in SNP_ind:
        _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
        _X = _X[gene_ind]  # get gene iid also in outcome iid
        _X -= _np.mean(_X)
        _[j] = _X @ _y / N / _np.var(_X)
    beta = _  # _np.sign(_)
    _pca = _y @ pca[gene_ind, :] / N / _np.var(pca[gene_ind, :], 0)
    beta = _np.hstack((_np.array([0.]), _pca, beta)).reshape(1, -1)
    #     beta = _np.sign(beta)
    beta_mat = _np.repeat(beta, len(lambda_) + 1, axis=0)
    for j in range(len(lambda_)):
        beta_mat[j + 1, :] = SNP_UAG_logistic_SCAD_MCP_PCA(
            bed_file=bed_file,
            bim_file=bim_file,
            fam_file=fam_file,
            outcome=outcome,
            SNP_ind=SNP_ind,
            L_convex=L_convex,
            pca=pca,
            beta_0=beta_mat[j, :],
            tol=tol,
            maxit=maxit,
            _lambda=lambda_[j],
            penalty=penalty,
            outcome_iid=outcome_iid,
            a=a,
            gamma=gamma)[1]
    return beta_mat[1:, :]


##############################################################################################
################ logsitic AG SNP bed-reader version with multiprocess ########################
##############################################################################################
# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)


def _SNP_update_smooth_grad_convex_logistic_parallel(N, SNP_ind, bed, beta_md,
                                                     y, outcome_iid, core_num,
                                                     multp):
    '''
    Update the gradient of the smooth convex objective component.
    '''
    p = bed.sid_count
    gene_iid = _np.array(list(bed.iid))
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]

    # first calcualte _=X@beta_md-y
    def __parallel_plus(_ind):
        import numpy as _np
        _X = bed.read(_np.s_[:, _ind],
                      dtype=_np.int8).flatten().reshape(-1, len(_ind))
        _X = _X[gene_ind, :]  # get gene iid also in outcome iid
        return _X @ beta_md[_ind + 1]
#         __ = _np.zeros(N)
#         for j in _ind:
#             _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
#             _X = _X[gene_ind]  # get gene iid also in outcome iid
#             __ += _X * beta_md[j + 1]
#         return __

# multiprocessing starts here

    _splited_array = _np.array_split(SNP_ind, core_num * multp)
    _splited_array = [
        __array for __array in _splited_array if __array.size != 0
    ]
    with _mp.Pool(core_num) as pl:
        _ = pl.map(__parallel_plus, _splited_array)
    _ = _np.array(_).sum(0)
    _ += beta_md[0]  # add the intercept
    _ = _np.tanh(_ / 2.) / 2. - _y + .5

    # then calculate _XTXbeta = X.T@X@beta_md = X.T@_
    def __parallel_assign(_ind):
        import numpy as _np
        _X = bed.read(_np.s_[:, _ind],
                      dtype=_np.int8).flatten().reshape(-1, len(_ind))
        _X = _X[gene_ind, :]  # get gene iid also in outcome iid
        return _ @ _X
#         k = 0
#         __ = _np.zeros(len(_ind))
#         for j in _ind:
#             _X = bed.read(_np.s_[:, j], dtype=_np.int8).flatten()
#             _X = _X[gene_ind]  # get gene iid also in outcome iid
#             __[k] = _X @ _
#             k += 1
#         return __

# multiprocessing starts here

    _splited_array = _np.array_split(SNP_ind, core_num * multp)
    _splited_array = [
        __array for __array in _splited_array if __array.size != 0
    ]
    with _mp.Pool(core_num) as pl:
        _XTXbeta = pl.map(__parallel_assign, _splited_array)
    __XTXbeta = _np.hstack(_XTXbeta)
    _XTXbeta = _np.zeros(p + 1)
    _XTXbeta[SNP_ind + 1] = __XTXbeta
    _XTXbeta[0] = _np.sum(_)
    del _
    del __XTXbeta
    return _XTXbeta / (2. * N)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_SCAD_logistic_parallel(N, SNP_ind, bed, beta_md, y,
                                                   outcome_iid, _lambda, a,
                                                   core_num, multp):
    '''
    Update the gradient of the smooth objective component for SCAD penalty.
    '''
    return _SNP_update_smooth_grad_convex_logistic_parallel(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid,
        core_num=core_num,
        multp=multp) + SCAD_concave_grad(x=beta_md, lambda_=_lambda, a=a)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_update_smooth_grad_MCP_logistic_parallel(N, SNP_ind, bed, beta_md, y,
                                                  outcome_iid, _lambda, gamma,
                                                  core_num, multp):
    '''
    Update the gradient of the smooth objective component for MCP penalty.
    '''
    return _SNP_update_smooth_grad_convex_logistic_parallel(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=beta_md,
        y=y,
        outcome_iid=outcome_iid,
        core_num=core_num,
        multp=multp) + MCP_concave_grad(
            x=beta_md, lambda_=_lambda, gamma=gamma)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def _SNP_lambda_max_logistic_parallel(bed, y, outcome_iid, N, SNP_ind,
                                      core_num, multp):
    """
    Calculate the lambda_max, i.e., the minimum lambda to nullify all penalized betas.
    """
    #     X_temp = X.copy()
    #     X_temp = X_temp[:,1:]
    #     X_temp -= _np.mean(X_temp,0).reshape(1,-1)
    #     X_temp /= _np.std(X_temp,0)
    #     y_temp = y.copy()
    #     y_temp -= _np.mean(y)
    #     y_temp /= _np.std(y)
    grad_at_0 = _SNP_update_smooth_grad_convex_logistic_parallel(
        N=N,
        SNP_ind=SNP_ind,
        bed=bed,
        beta_md=_np.zeros(len(SNP_ind)),
        y=y,
        outcome_iid=outcome_iid,
        core_num=core_num,
        multp=multp)
    return _np.linalg.norm(grad_at_0[1:], ord=_np.infty)


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_UAG_logistic_SCAD_MCP_parallel(bed_file,
                                       bim_file,
                                       fam_file,
                                       outcome,
                                       outcome_iid,
                                       SNP_ind,
                                       L_convex,
                                       beta_0=_np.ones(1),
                                       tol=1e-5,
                                       maxit=500,
                                       _lambda=.5,
                                       penalty="SCAD",
                                       a=3.7,
                                       gamma=2.,
                                       core_num="NOT DECLARED",
                                       multp=1):
    '''
    Carry out the optimization for penalized logistic for a fixed lambda.
    '''
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    y = outcome
    p = bed.sid_count
    gene_iid = _np.array(list(bed.iid))
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    if _np.all(beta_0 == _np.ones(1)):
        _y = y[_np.intersect1d(outcome_iid,
                               gene_iid,
                               assume_unique=True,
                               return_indices=True)[1]]
        _y = _y.astype(dtype=_np.float64)
        _y -= _np.mean(_y)

        def __parallel_assign(_ind):
            import numpy as _np
            _X = bed.read(_np.s_[:, _ind],
                          dtype=_np.float64).flatten().reshape(-1, len(_ind))
            _X = _X[gene_ind, :]  # get gene iid also in outcome iid
            _X -= _np.mean(_X, 0).reshape(1, -1)
            return _y @ _X / N
#             k = 0
#             __ = _np.zeros(len(_ind))
#             for j in _ind:
#                 _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
#                 _X = _X[gene_ind]  # get gene iid also in outcome iid
#                 _X -= _np.mean(_X)
#                 __[k] = _X @ _y / N
#                 k += 1
#             return __

# multiprocessing starts here

        _splited_array = _np.array_split(SNP_ind, core_num * multp)
        _splited_array = [
            __array for __array in _splited_array if __array.size != 0
        ]
        with _mp.Pool(core_num) as pl:
            _XTy = pl.map(__parallel_assign, _splited_array)
        _XTy = _np.hstack(_XTy)
        beta = _np.zeros(p + 1)
        beta[SNP_ind + 1] = _np.sign(_XTy)
    else:
        beta = beta_0
    # passing other parameters
    smooth_grad = _np.ones(p + 1)
    beta_ag = beta.copy()
    beta_md = beta.copy()
    k = 0
    converged = False
    opt_alpha = 1.
    old_speed_norm = 1.
    speed_norm = 1.
    restart_k = 0

    if penalty == "SCAD":
        #         L = _np.max(_np.array([L_convex, 1./(a-1)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (a - 1)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_SCAD_logistic_parallel(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                a=a,
                core_num=core_num,
                multp=multp)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    else:
        #         L = _np.max(_np.array([L_convex, 1./(gamma)]))
        L = _np.linalg.norm(_np.array([L_convex, 1. / (gamma)]), ord=_np.infty)
        opt_beta = .99 / L
        while ((not converged) or (k < 3)) and k <= maxit:
            k += 1
            if old_speed_norm > speed_norm and k - restart_k >= 3:  # in this case, restart
                opt_alpha = 1.  # restarting
                restart_k = k  # restarting
            else:  # restarting
                opt_alpha = 2 / (
                    1 + (1 + 4. / opt_alpha**2)**.5
                )  # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            # parameter settings based on minimizing Ghadimi and Lan's rate of convergence error upper bound
            opt_lambda = opt_beta / opt_alpha
            beta_md_old = beta_md.copy()  # restarting
            beta_md = (1 - opt_alpha) * beta_ag + opt_alpha * beta
            old_speed_norm = speed_norm  # restarting
            speed_norm = _np.linalg.norm(beta_md - beta_md_old,
                                         ord=2)  # restarting
            converged = (_np.linalg.norm(beta_md - beta_md_old, ord=_np.infty)
                         < tol)
            smooth_grad = _SNP_update_smooth_grad_MCP_logistic_parallel(
                N=N,
                SNP_ind=SNP_ind,
                bed=bed,
                beta_md=beta_md,
                y=y,
                outcome_iid=outcome_iid,
                _lambda=_lambda,
                gamma=gamma,
                core_num=core_num,
                multp=multp)
            beta = soft_thresholding(x=beta - opt_lambda * smooth_grad,
                                     lambda_=opt_lambda * _lambda)
            beta_ag = soft_thresholding(x=beta_md - opt_beta * smooth_grad,
                                        lambda_=opt_beta * _lambda)
#             converged = _np.all(_np.max(_np.abs(beta_md - beta_ag)/opt_beta) < tol).item()
#             converged = (_np.linalg.norm(beta_md - beta_ag, ord=_np.infty) < (tol*opt_beta))
    return k, beta_md


# @_jit(nopython=True, cache=True, parallel=True, fastmath=True, nogil=True)
def SNP_solution_path_logistic_parallel(bed_file,
                                        bim_file,
                                        fam_file,
                                        outcome,
                                        outcome_iid,
                                        lambda_,
                                        L_convex,
                                        SNP_ind,
                                        beta_0=_np.ones(1),
                                        tol=1e-5,
                                        maxit=500,
                                        penalty="SCAD",
                                        a=3.7,
                                        gamma=2.,
                                        core_num="NOT DECLARED",
                                        multp=1):
    '''
    Carry out the optimization for the solution path without the strong rule.
    '''
    if core_num == "NOT DECLARED":
        core_num = _mp.cpu_count()
    else:
        assert core_num <= _mp.cpu_count(
        ), "Declared number of cores used for multiprocessing should not exceed number of cores on this machine."
    assert core_num >= 2, "Multiprocessing should not be used on single-core machines."

    bed = _open_bed(filepath=bed_file,
                    fam_filepath=fam_file,
                    bim_filepath=bim_file)
    y = outcome
    p = bed.sid_count
    gene_iid = _np.array(list(bed.iid))
    gene_ind = _np.intersect1d(gene_iid,
                               outcome_iid,
                               assume_unique=True,
                               return_indices=True)[1]
    N = len(
        _np.intersect1d(outcome_iid,
                        gene_iid,
                        assume_unique=True,
                        return_indices=True)[1])
    _y = y[_np.intersect1d(outcome_iid,
                           gene_iid,
                           assume_unique=True,
                           return_indices=True)[1]]
    _y = _y.astype(dtype=_np.float64)
    _y -= _np.mean(_y)

    def __parallel_assign(_ind):
        import numpy as _np
        _X = bed.read(_np.s_[:, _ind],
                      dtype=_np.float64).flatten().reshape(-1, len(_ind))
        _X = _X[gene_ind, :]  # get gene iid also in outcome iid
        _X -= _np.mean(_X, 0).reshape(1, -1)
        return _y @ _X / N


#         k = 0
#         __ = _np.zeros(len(_ind))
#         for j in _ind:
#             _X = bed.read(_np.s_[:, j], dtype=_np.float64).flatten()
#             _X = _X[gene_ind]  # get gene iid also in outcome iid
#             _X -= _np.mean(_X)
#             __[k] = _X @ _y / N
#             k += 1
#         return __

# multiprocessing starts here

    _splited_array = _np.array_split(SNP_ind, core_num * multp)
    _splited_array = [
        __array for __array in _splited_array if __array.size != 0
    ]
    with _mp.Pool(core_num) as pl:
        _XTy = pl.map(__parallel_assign, _splited_array)
    _XTy = _np.hstack(_XTy)
    beta = _np.zeros(p + 1)
    beta[SNP_ind + 1] = _XTy  # _np.sign(_XTy)
    beta = beta.reshape(1, -1)

    beta_mat = _np.zeros((len(lambda_) + 1, p + 1))
    beta_mat = _np.repeat(beta, len(lambda_) + 1, axis=0)
    for j in range(len(lambda_)):
        beta_mat[j + 1, :] = SNP_UAG_logistic_SCAD_MCP_parallel(
            bed_file=bed_file,
            bim_file=bim_file,
            fam_file=fam_file,
            outcome=outcome,
            SNP_ind=SNP_ind,
            L_convex=L_convex,
            beta_0=beta_mat[j, :],
            tol=tol,
            maxit=maxit,
            _lambda=lambda_[j],
            penalty=penalty,
            outcome_iid=outcome_iid,
            a=a,
            gamma=gamma,
            core_num=core_num,
            multp=multp)[1]
    return beta_mat[1:, :]
