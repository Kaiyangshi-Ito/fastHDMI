import os

import fastHDMI as mi
# from dask import dataframe as dd
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, rankdata
from sklearn.linear_model import (ElasticNetCV, LarsCV, LassoCV, LassoLarsCV,
                                  LogisticRegressionCV, RidgeCV)
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler
from tqdm import tqdm

case_outlier_csv = os.environ["SLURM_TMPDIR"] + \
    r"/df_outlier_asd.csv"
case_outlier_df = pd.read_csv(case_outlier_csv,
                              encoding="unicode_escape",
                              engine="c")
case_outlier_subid_ind = (case_outlier_df["outlier_ind"] == 1)
case_outlier_subid = case_outlier_df["subId"][case_outlier_subid_ind]

csv_file = os.environ["SLURM_TMPDIR"] + \
    r"/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
abide = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")

# create an index to preserve
extract_ind = (abide["DX_GROUP"] == 1)  # extract only the cases

# reserve only the cases
_abide = abide[extract_ind]

# drop the outliers
_abide = _abide[~_abide["SUB_ID"].isin(case_outlier_subid)]

# write to another csv
_abide.to_csv(
    r"/home/kyang/projects/def-cgreenwo/kyang/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX_CasesOnly.csv"
)
