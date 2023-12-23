import numpy as np
import pandas as pd
# from dask import dataframe as dd
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
import itertools

# read the data
csv_file = os.environ["SLURM_TMPDIR"] + \
    r"/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"

abide_original = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")
_abide_name = abide_original.columns.tolist()[1:]

# print(_abide_name)

abide_name_original = _abide_name[1:-3]

# preserve only the neuro-imaging data
abide_original = abide_original[abide_name_original]


def convert2list(a):
    b = np.asarray(a)
    return b.tolist()


def sim_based_on_abide_continuous(pair):
    abide, abide_name = abide_original.copy(), abide_name_original.copy()
    _num_true_vars, _seed = pair
    SNR = 3.
    num_true_vars = _num_true_vars
    seed = _seed
    assert num_true_vars < len(abide_name)
    np.random.seed(seed)

    true_attr_label = np.zeros(len(abide_name), dtype=int)
    true_attr_index = np.arange(len(abide_name))
    true_attr_index = np.random.choice(true_attr_index,
                                       num_true_vars,
                                       replace=False)
    true_names = np.take(abide_name,
                         true_attr_index)  # this is a list for true names
    true_names = convert2list(true_names)
    true_attr_label[
        true_attr_index] = 1  # true_attr_label is binary indicate whether the covaraite is "true"

    true_beta = np.random.choice([1., -1.], num_true_vars, replace=True)
    #     true_beta = np.random.uniform(low=5.0, high=6.0,
    #                                   size=num_true_vars) * np.random.choice(
    #                                       [1., -1.], num_true_vars, replace=True)

    sim_data = abide[true_names].to_numpy(copy=True)
    sim_data = StandardScaler(copy=False).fit_transform(sim_data)
    sim_data = sim_data**2
    sim_data = StandardScaler(copy=False).fit_transform(sim_data)

    X_cov = np.corrcoef(sim_data, rowvar=False)
    true_sigma_sim = np.sqrt(true_beta.T @ X_cov @ true_beta / SNR)

    outcome = sim_data @ true_beta + np.random.normal(0, true_sigma_sim,
                                                      sim_data.shape[0])

    abide["outcome"] = outcome
    abide = abide[["outcome"] + abide_name]

    print("The outcome is continuous.")

    print("Our developed FFT-based MI calculation:")

    try:
        mi_output = mi.continuous_screening_dataframe_parallel(
            dataframe=abide,
            _usecols=["outcome"] + abide_name,
            multp=10,
            core_num=32,
            share_memory=False,
            kernel="epa",
            bw="ISJ",
            norm=2)
    except:
        mi_output = mi.continuous_screening_dataframe_parallel(
            dataframe=abide,
            _usecols=["outcome"] + abide_name,
            multp=10,
            core_num=32,
            share_memory=False,
            kernel="epa",
            bw="silverman",
            norm=2)

    print("Our developed binning MI calculation:")

    binning_mi_output = mi.binning_continuous_screening_dataframe_parallel(
        dataframe=abide,
        _usecols=["outcome"] + abide_name,
        multp=10,
        core_num=32,
        share_memory=False)

    print("sklearn MI calculation:")

    skmi_output = mi.continuous_skMI_screening_dataframe_parallel(
        dataframe=abide,
        _usecols=["outcome"] + abide_name,
        multp=10,
        core_num=32,
        random_state=0,
        share_memory=False)

    print("Pearson's correlation calculation:")

    pearson_output = mi.Pearson_screening_dataframe_parallel(
        dataframe=abide,
        _usecols=["outcome"] + abide_name,
        multp=10,
        core_num=32,
        share_memory=False)

    mi_pseudo_prob = np.abs(mi_output) / np.max(np.abs(mi_output))
    mi_auroc = roc_auc_score(true_attr_label, mi_pseudo_prob)
    binning_mi_pseudo_prob = np.abs(binning_mi_output) / np.max(
        np.abs(binning_mi_output))
    binning_mi_auroc = roc_auc_score(true_attr_label, binning_mi_pseudo_prob)
    skmi_pseudo_prob = np.abs(skmi_output) / np.max(np.abs(skmi_output))
    skmi_auroc = roc_auc_score(true_attr_label, skmi_pseudo_prob)
    pearson_pseudo_prob = np.abs(pearson_output) / np.max(
        np.abs(pearson_output))
    pearson_auroc = roc_auc_score(true_attr_label, pearson_pseudo_prob)

    del mi_output, binning_mi_output, skmi_output, pearson_output, abide, abide_name, true_names, true_beta, sim_data, X_cov, true_sigma_sim, outcome, mi_pseudo_prob, skmi_pseudo_prob, pearson_pseudo_prob

    return np.array([mi_auroc, skmi_auroc, pearson_auroc, binning_mi_auroc])


num_true_vars_list = [120]
seed_list = range(100)

itrs = itertools.product(num_true_vars_list, seed_list)

output_array = np.array(list(map(sim_based_on_abide_continuous, tqdm(itrs))))
output_array = output_array.reshape(1, 100, 4).squeeze()
np.save(r"./ABIDE_continuous_120", output_array)
