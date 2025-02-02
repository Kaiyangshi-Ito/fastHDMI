import os

import fastHDMI as mi
# from dask import dataframe as dd
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pandas as pd
from scipy.linalg import block_diag, toeplitz
from scipy.stats import kendalltau, norm, rankdata
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (ElasticNetCV, LarsCV, LassoCV, LassoLarsCV,
                                  LinearRegression, LogisticRegression,
                                  LogisticRegressionCV, RidgeCV)
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler
from tqdm import tqdm

csv_file = os.environ["SLURM_TMPDIR"] + \
    r"/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
abide = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")
# abide = dd.read_csv(csv_file, sample=1250000)

_abide_name = abide.columns.tolist()[1:]
# _abide_name = list(abide.columns)[1:]

# print(_abide_name)

# we don't inlcude covariates for adjustment in the screening since we choose to always include them in the model

abide_name = [_abide_name[-1]] + _abide_name[1:-3]
# so that the left first column is the outcome and the rest columns are areas

del _abide_name

print("The outcome is diagnosis.")
print("Now running using c CSV engine with share_memory=False.")
print("Our developed FFT-based MI calculation:")

for _kernel in [
        'gaussian', 'exponential', 'box', 'tri', 'epa', 'biweight',
        'triweight', 'tricube', 'cosine'
]:
    for _bw in ['silverman', 'scott', 'ISJ']:
        try:
            mi_output = mi.binary_screening_csv_parallel(
                csv_file,
                _usecols=abide_name.copy(),
                csv_engine="c",
                sample=1250000,
                multp=10,
                core_num=16,
                share_memory=False,
                kernel=_kernel,
                bw=_bw)
            if "high_mem" == "high_mem":
                np.save(
                    r"./ABIDE_diagnosis_MI_{kernel}_{bw}_output".format(
                        kernel=_kernel, bw=_bw), mi_output)

            del mi_output

        except:
            print("This kernel-bw combination reports an error: ", _kernel,
                  _bw)

print("binning MI calculation:")

binning_output = mi.binning_binary_screening_csv_parallel(
    csv_file,
    _usecols=abide_name.copy(),
    csv_engine="c",
    sample=1250000,
    multp=10,
    core_num=16,
    share_memory=False)
if "high_mem" == "high_mem":
    np.save(r"./ABIDE_diagnosis_binningMI_output", binning_output)

print("sklearn MI calculation:")

skmi_output = mi.binary_skMI_screening_csv_parallel(csv_file,
                                                    _usecols=abide_name.copy(),
                                                    csv_engine="c",
                                                    sample=1250000,
                                                    multp=10,
                                                    core_num=16,
                                                    random_state=0,
                                                    share_memory=False)
if "high_mem" == "high_mem":
    np.save(r"./ABIDE_diagnosis_skMI_output", skmi_output)

del skmi_output

print("Pearson's correlation calculation:")

pearson_output = mi.Pearson_screening_csv_parallel(csv_file,
                                                   _usecols=abide_name.copy(),
                                                   csv_engine="c",
                                                   sample=1250000,
                                                   multp=10,
                                                   core_num=16,
                                                   share_memory=False)
if "high_mem" == "high_mem":
    np.save(r"./ABIDE_diagnosis_Pearson_output", pearson_output)

del pearson_output
