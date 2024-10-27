# running PCA on it

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
    r"/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
abide = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")
_sex = abide["SEX"].tolist()
_age = abide["AGE_AT_SCAN"].tolist()
_diagnosis = abide["DX_GROUP"].tolist()
abide = abide.iloc[:, 2:-3].to_numpy(copy=False)
abide = StandardScaler(copy=False).fit_transform(abide)

pca = PCA(n_components=abide.shape[0])
components = pca.fit(
    abide
).components_.T  # transpose because it's a ndarray of shape (n_components, n_features)
pca_output = abide @ components
cols = ["PC_" + str(j + 1) for j in range(pca_output.shape[1])]
df = pd.DataFrame(pca_output, columns=cols)
df["SEX"] = _sex
df["AGE_AT_SCAN"] = _age
df["DX_GROUP"] = _diagnosis
df.to_csv(r"./ABIDE_PCA/ABIDE_PCA.csv")
