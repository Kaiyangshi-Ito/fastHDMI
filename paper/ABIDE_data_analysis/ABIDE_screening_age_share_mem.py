import numpy as np
import pandas as pd
from dask import dataframe as dd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, rankdata, norm
from scipy.linalg import block_diag, toeplitz
import fastHDMI as mi
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV, LarsCV, LassoLarsCV, LogisticRegressionCV, LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score
import multiprocess as mp
from tqdm import tqdm
import os


csv_file = os.environ["SLURM_TMPDIR"] + \
    r"/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX_CasesOnly.csv"
abide = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")
# abide = dd.read_csv(csv_file, sample=1250000)

_abide_name = abide.columns.tolist()[1:]
# _abide_name = list(abide.columns)[1:]

# print(_abide_name)

# we don't inlcude covariates for adjustment in the screening since we choose to always include them in the model

abide_name = [_abide_name[-3]] + _abide_name[1:-3]
# so that the left first column is the outcome and the rest columns are areas

np.save(r"./ABIDE_columns", _abide_name[1:-3])

del _abide_name

print("The outcome is age.")
print("Now running using c CSV engine with share_memory=True.")
print("Our developed FFT-based MI calculation:")

for _kernel in [
        'gaussian', 'exponential', 'box', 'tri', 'epa', 'biweight',
        'triweight', 'tricube', 'cosine'
]:
    for _bw in ['silverman', 'scott', 'ISJ']:
        try:
            mi_output = mi.continuous_screening_csv_parallel(
                csv_file,
                _usecols=abide_name.copy(),
                csv_engine="c",
                sample=1250000,
                multp=10,
                core_num=16,
                share_memory=True,
                kernel=_kernel,
                bw=_bw,
                norm=2)
            if "share_mem" == "high_mem":
                np.save(
                    r"./ABIDE_age_MI_{kernel}_{bw}_output".format(
                        kernel=_kernel, bw=_bw), mi_output)

            del mi_output

        except:
            print("This kernel-bw combination reports an error: ", _kernel,
                  _bw)

print("binning MI calculation:")

binning_output = mi.binning_continuous_screening_csv_parallel(
    csv_file,
    _usecols=abide_name.copy(),
    csv_engine="c",
    sample=1250000,
    multp=10,
    core_num=16,
    share_memory=True)
if "share_mem" == "high_mem":
    np.save(r"./ABIDE_age_binningMI_output", binning_output)

print("sklearn MI calculation:")

skmi_output = mi.continuous_skMI_screening_csv_parallel(
    csv_file,
    _usecols=abide_name.copy(),
    csv_engine="c",
    sample=1250000,
    multp=10,
    core_num=16,
    random_state=0,
    share_memory=True)
if "share_mem" == "high_mem":
    np.save(r"./ABIDE_age_skMI_output", skmi_output)

del skmi_output

print("Pearson's correlation calculation:")

pearson_output = mi.Pearson_screening_csv_parallel(csv_file,
                                                   _usecols=abide_name.copy(),
                                                   csv_engine="c",
                                                   sample=1250000,
                                                   multp=10,
                                                   core_num=16,
                                                   share_memory=True)
if "share_mem" == "high_mem":
    np.save(r"./ABIDE_age_Pearson_output", pearson_output)

del pearson_output
