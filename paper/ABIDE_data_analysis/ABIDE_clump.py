import os

import fastHDMI as mi
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pandas as pd
from dask import dataframe as dd
from scipy.stats import kendalltau, norm, rankdata
from sklearn.linear_model import (ElasticNetCV, LarsCV, LassoCV, LassoLarsCV,
                                  LogisticRegressionCV, RidgeCV)
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler
from tqdm import tqdm

csv_file = os.environ["SLURM_TMPDIR"] + \
    r"/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
abide = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")

_abide_name = abide.columns.tolist()[1:]

# Create correlation matrix
corr_matrix = abide.iloc[:, 3:5000].corr().abs()

np.save(r"./ABIDE_corr3,5000", corr_matrix)

del corr_matrix

corr_matrix = abide.iloc[:, 50000:55000].corr().abs()

np.save(r"./ABIDE_corr50000,55000", corr_matrix)

del corr_matrix

corr_matrix = abide.iloc[:, 90000:95000].corr().abs()

np.save(r"./ABIDE_corr90000,95000", corr_matrix)

# Select upper triangle of correlation matrix
upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
# df.drop(to_drop, axis=1, inplace=True)
