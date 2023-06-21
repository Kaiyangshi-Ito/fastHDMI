import numpy as np
import pandas as pd
from dask import dataframe as dd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, rankdata, norm
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
    r"/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
original_df = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")

top_colnames_num = 5

csv_file = os.environ["SLURM_TMPDIR"] + \
    r"/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
original_df = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")

columns = np.load(os.environ["SLURM_TMPDIR"] + r"/ABIDE_columns.npy")

for outcome in ["diagnosis", "age"]:
    top_colnames = []
    for dep_measure in ["MI_epa_ISJ", "Pearson", "skMI"]:
        abide_dep = np.load(os.environ["SLURM_TMPDIR"] + r"/ABIDE_" + outcome +
                            r"_" + dep_measure + r"_output.npy")
        abide_dep = np.absolute(abide_dep)

        top_colnames = np.hstack(
            (top_colnames, columns[np.argsort(-abide_dep)][:top_colnames_num]))

    top_colnames = list(set(top_colnames))
    for colname in top_colnames:
        if outcome == "diagnosis":
            plt.scatter(original_df["DX_GROUP"],
                        original_df[colname],
                        alpha=.2)
        if outcome == "age":
            plt.scatter(original_df["AGE_AT_SCAN"],
                        original_df[colname],
                        alpha=.2)
        plt.ylabel(outcome)
        plt.xlabel(colname)
        plt.title(
            r"scatter plot for outcome vs the top associated covariates: " +
            outcome + r" and " + colname)
        plt.savefig(r"scatter_" + outcome + r"_" + colname, format="eps")
        plt.close()
