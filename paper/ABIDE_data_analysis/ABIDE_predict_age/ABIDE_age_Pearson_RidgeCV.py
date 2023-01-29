import numpy as np
import pandas as pd
from dask import dataframe as dd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import rankdata
import fastHDMI as mi
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import multiprocess as mp

csv_file = r"/home/kyang/projects/def-cgreenwo/abide_data/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
original_df = pd.read_csv(csv_file, encoding='unicode_escape', engine='c')


def testing_error(num_covariates=20,
                  training_proportion=.8,
                  fun=ElasticNetCV,
                  outcome_name="AGE_AT_SCAN",
                  seed=1):
    np.random.seed(seed)
    columns = np.load(r"./ABIDE_columns.npy")
    abide_mi = np.load(r"./ABIDE_age_Pearson_output.npy"
                       )  # for Pearson, use ABIDE_age_Pearson_output.npy
    _usecols = np.hstack((outcome_name, "SEX", "DX_GROUP",
                          columns[np.argsort(-abide_mi)][:num_covariates]))

    df = original_df[_usecols].dropna(inplace=False).sample(
        frac=1, random_state=seed, replace=False).reset_index(drop=True,
                                                              inplace=False)
    if df.shape[0] > 20:
        train_test_div = int(np.around(df.shape[0] * training_proportion))
        X_train, y_train = df.iloc[:train_test_div,
                                   1:], df.iloc[:train_test_div, 0]
        X_test, y_test = df.iloc[train_test_div:, 1:], df.iloc[train_test_div:,
                                                               0]
        if fun in [ElasticNetCV, LassoCV]:
            fit = fun(cv=5, random_state=seed).fit(X_train, y_train)
        elif fun in [RidgeCV]:  # RidgeCV doesn't have seed setting
            fit = fun(cv=5).fit(X_train, y_train)
        y_pred = fit.predict(X_test)
        out = r2_score(y_test, y_pred)
    else:
        out = np.nan
    return out


def testing_error_rep(num_covariates=20,
                      training_proportion=.8,
                      fun=ElasticNetCV,
                      outcome_name="AGE_AT_SCAN",
                      num_rep=10):
    def _testing_error(seed):
        return testing_error(num_covariates=num_covariates,
                             training_proportion=training_proportion,
                             fun=fun,
                             outcome_name=outcome_name,
                             seed=seed)

    seeds = np.arange(num_rep)
    return np.array(list(map(_testing_error, seeds)))


def testing_error_num_attr(num_attr,
                           training_proportion=.8,
                           fun=ElasticNetCV,
                           outcome_name="AGE_AT_SCAN",
                           num_rep=10):
    def _testing_error_rep(_num_attr):
        return testing_error_rep(num_covariates=_num_attr,
                                 training_proportion=training_proportion,
                                 fun=fun,
                                 outcome_name=outcome_name,
                                 num_rep=num_rep)

    return np.array(list(map(_testing_error_rep, num_attr)))


output = testing_error_num_attr(
    num_attr=list(
        map(int,
            np.around(np.linspace(0, 149959 / 100, 50 + 1)[1:]).tolist())
    ),  # so here it will screen the number of covariates roughly 30 apart
    training_proportion=.8,  # 80/20 training+validation/testing division
    fun=RidgeCV,  # here it says to use ElasticNetCV
    outcome_name="AGE_AT_SCAN",
    num_rep=200)
np.save(r"./ABIDE_age_Pearson_RidgeCV", output)