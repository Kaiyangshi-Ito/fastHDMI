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


def sim_based_on_abide_binary(pair):
    abide, abide_name = abide_original.copy(), abide_name_original.copy()
    _num_true_vars, _seed = pair
    SNR = 3.
    num_true_vars = _num_true_vars
    seed = _seed
    assert num_true_vars < len(abide_name)
    np.random.seed(seed)

    true_names = np.random.choice(abide_name, num_true_vars, replace=False)
    true_names = convert2list(true_names)

    true_beta = np.random.uniform(low=1.0, high=2.0,
                                  size=num_true_vars) * np.random.choice(
                                      [1., -1.], num_true_vars, replace=True)

    sim_data = abide[true_names].to_numpy(copy=True)
    sim_data = StandardScaler(copy=False).fit_transform(sim_data)
    signal = sim_data @ true_beta

    outcome = np.random.binomial(1, np.tanh(signal / 2) / 2 + .5)

    abide["outcome"] = outcome
    abide = abide[["outcome"] + abide_name]

    print("The outcome is binary.")

    print("Our developed FFT-based MI calculation:")

    try:
        mi_output = mi.binary_screening_dataframe_parallel(
            dataframe=abide,
            _usecols=["outcome"] + abide_name,
            multp=10,
            core_num=32,
            share_memory=False,
            kernel="epa",
            bw="ISJ")
    except:
        print("The kernel-bw combination reports an error.")

    print("sklearn MI calculation:")

    skmi_output = mi.binary_skMI_screening_dataframe_parallel(
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

    mi_selection = np.asarray(abide_name)[np.argsort(
        -mi_output)][:num_true_vars]
    mi_selection = convert2list(mi_selection)
    skmi_selection = np.asarray(abide_name)[np.argsort(
        -skmi_output)][:num_true_vars]
    skmi_selection = convert2list(skmi_selection)
    pearson_selection = np.asarray(abide_name)[np.argsort(
        -pearson_output)][:num_true_vars]
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


num_true_vars_list = [30000]
seed_list = range(100)

itrs = itertools.product(num_true_vars_list, seed_list)

output_array = np.array(list(map(sim_based_on_abide_binary, tqdm(itrs))))
output_array = output_array.reshape(1, 100, 3).squeeze()
np.save(r"./ABIDE_simulations/ABIDE_sim_binary_30000", output_array)
