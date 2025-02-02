import os

import fastHDMI as mi
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pandas as pd
from dask import dataframe as dd
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
    r"/ABIDE_PCA.csv"
# abide = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")
abide = dd.read_csv(csv_file, sample=1250000)

# _abide_name = abide.columns.tolist()[1:]
_abide_name = list(abide.columns)[1:]

# print(_abide_name)

# we don't inlcude age and sex in the screening since we choose to always include them in the model

abide_name = [_abide_name[-2]] + _abide_name[:-3]
# so that the left first column is the outcome and the rest columns are areas

np.save(r"./ABIDE_columns", _abide_name[:-3])

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
