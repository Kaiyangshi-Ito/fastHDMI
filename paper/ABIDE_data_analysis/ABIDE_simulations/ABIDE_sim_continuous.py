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
import itertools


def convert2list(a):
    b = np.asarray(a)
    return b.tolist()


def sim_based_on_abide_continuous(pair):
    # read the data
    csv_file = os.environ["SLURM_TMPDIR"] + \
        r"/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"

    abide = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")
    _abide_name = abide.columns.tolist()[1:]

    # print(_abide_name)
    _num_true_vars, _seed = pair
    abide_name = _abide_name[1:-3]

    # preserve only the neuro-imaging data
    abide = abide[abide_name]

    SNR = 3.
    num_true_vars = _num_true_vars
    seed = _seed
    assert num_true_vars < len(abide_name)

    true_names = np.random.choice(abide_name, num_true_vars, replace=False)
    true_names = convert2list(true_names)

    true_beta = np.random.uniform(low=1.0, high=2.0,
                                  size=num_true_vars) * np.random.choice(
                                      [1., -1.], num_true_vars, replace=True)

    sim_data = abide[true_names].to_numpy(copy=True)
    sim_data = StandardScaler(copy=False).fit_transform(sim_data)
    sim_data += np.abs(
        np.min(sim_data, 1).reshape(1, -1)
    )  # to ensure that all the the data are positive so we can take square root
    sim_data = np.sqrt(sim_data)
    signs = np.random.choice([1., -1.], sim_data.size,
                             replace=True).reshape(sim_data.shape)
    sim_data = sim_data * signs
    sim_data = StandardScaler(copy=False).fit_transform(sim_data)

    X_cov = np.corrcoef(sim_data)
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
            _usecols=abide_name.copy(),
            csv_engine="c",
            sample=1250000,
            multp=10,
            core_num=10,
            share_memory=False,
            kernel="epa",
            bw="ISJ",
            norm=2)
    except:
        print("The kernel-bw combination reports an error.")

    print("sklearn MI calculation:")

    skmi_output = mi.continuous_skMI_screening_dataframe_parallel(
        dataframe=abide,
        _usecols=abide_name.copy(),
        csv_engine="c",
        sample=1250000,
        multp=10,
        core_num=10,
        random_state=0,
        share_memory=False)

    print("Pearson's correlation calculation:")

    pearson_output = mi.Pearson_screening_dataframe_parallel(
        dataframe=abide,
        _usecols=abide_name.copy(),
        csv_engine="c",
        sample=1250000,
        multp=10,
        core_num=10,
        share_memory=False)

    mi_selection = abide_name[np.argsort(-mi_output)][:num_true_vars]
    mi_selection = convert2list(mi_selection)
    skmi_selection = abide_name[np.argsort(-skmi_output)][:num_true_vars]
    skmi_selection = convert2list(skmi_selection)
    pearson_selection = abide_name[np.argsort(-pearson_output)][:num_true_vars]
    pearson_selection = convert2list(pearson_selection)

    mi_sensitivity = len(set(mi_selection)) + len(set(true_names)) - len(
        set(mi_selection + true_names))
    mi_sensitivity = mi_sensitivity / len(true_names)
    skmi_sensitivity = len(set(skmi_selection)) + len(set(true_names)) - len(
        set(skmi_selection + true_names))
    skmi_sensitivity = skmi_sensitivity / len(true_names)
    pearson_sensitivity = len(set(pearson_selection)) + len(
        set(true_names)) - len(set(pearson_selection + true_names))
    pearson_sensitivity = pearson_sensitivity / len(true_names)
    return np.array([mi_sensitivity, skmi_sensitivity, pearson_sensitivity])


num_true_vars_list = list(map(int, np.linspace(0, 50000,
                                               6)))[1:]  # we don't want 0 here
seed_list = range(100)

itrs = itertools.product(num_true_vars_list, seed_list)

output_array = np.array(list(map(sim_based_on_abide_continuous, tqdm(itrs))))
output_array = output_array.reshape(5, 100, 3)
np.save(r"./ABIDE_simulations/ABIDE_sim_continuous", output_array)