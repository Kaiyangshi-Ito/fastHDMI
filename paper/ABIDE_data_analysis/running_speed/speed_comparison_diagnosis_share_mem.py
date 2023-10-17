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
import timeit

csv_file = os.environ["SLURM_TMPDIR"] + \
    r"/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
abide = pd.read_csv(csv_file, encoding="unicode_escape", engine="c")
# abide = dd.read_csv(csv_file, sample=1250000)

_abide_name = abide.columns.tolist()[1:]
# _abide_name = list(abide.columns)[1:]

# print(_abide_name)

# we don't inlcude age and sex in the screening since we choose to always include them in the model

# diagnosis
abide_name = [_abide_name[-1]] + _abide_name[1:-3]
# so that the left first column is the outcome and the rest columns are areas
del _abide_name

num_input_vars_divide = 10
prop_input_vars_list = np.linspace(0, 1, num_input_vars_divide)[1:]


def _get_computing_time(prop_input_vars):
    s = '''mi.binary_screening_csv_parallel(csv_file,_usecols=abide_name.copy()[0:int(len(abide_name)*prop_input_vars)],csv_engine="c",sample=1250000,multp=10,core_num=16,share_memory=True,kernel="epa",bw="ISJ",verbose=0)'''
    imports_and_vars = globals()
    imports_and_vars.update(locals())
    num_loops = timeit.Timer(stmt=s, globals=imports_and_vars).autorange()[0]
    FFTKDE_MI_times = timeit.Timer(stmt=s, globals=imports_and_vars).repeat(
        repeat=5, number=num_loops)

    s = '''mi.binary_skMI_screening_csv_parallel(csv_file,_usecols=abide_name.copy()[0:int(len(abide_name)*prop_input_vars)],csv_engine="c",sample=1250000,multp=10,core_num=16,random_state=0,share_memory=True,verbose=0)'''
    imports_and_vars = globals()
    imports_and_vars.update(locals())
    num_loops = timeit.Timer(stmt=s, globals=imports_and_vars).autorange()[0]
    sklearn_MI_times = timeit.Timer(stmt=s, globals=imports_and_vars).repeat(
        repeat=5, number=num_loops)

    s = '''pearson_output = mi.Pearson_screening_csv_parallel( csv_file, _usecols=abide_name.copy()[0:int(len(abide_name)*prop_input_vars)], csv_engine="c", sample=1250000, multp=10, core_num=16, share_memory=True,verbose=0)'''
    imports_and_vars = globals()
    imports_and_vars.update(locals())
    num_loops = timeit.Timer(stmt=s, globals=imports_and_vars).autorange()[0]
    Pearson_times = timeit.Timer(stmt=s, globals=imports_and_vars).repeat(
        repeat=5, number=num_loops)

    return np.vstack((FFTKDE_MI_times, sklearn_MI_times, Pearson_times))


print(
    "Running speed comparison when the outcome variable is diagnosis (binary) and the memory setting is share_mem... "
)
output_array = np.array(
    list(map(_get_computing_time, tqdm(prop_input_vars_list))))
np.save(r"./running_time_diagnosis_share_mem", output_array)