import numpy as np
import pandas as pd
from dask import dataframe as dd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from scipy.stats import rankdata
from scipy.stats import norm
import fastHDMI as mi
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LarsCV
from sklearn.linear_model import LassoLarsCV
from sklearn.metrics import r2_score
import multiprocess as mp
from tqdm import tqdm as tqdm
import os

csv_file = r"/home/kyang/projects/def-cgreenwo/abide_data/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
# abide = pd.read_csv(csv_file, encoding='unicode_escape', engine="c")
abide = dd.read_csv(csv_file, sample=1250000)

# _abide_name = abide.columns.tolist()[1:]
_abide_name = list(abide.columns)[1:]

# print(_abide_name)

# we don't inlcude age and sex in the screening since they should always be included in the model
abide_name = [_abide_name[-1]] + _abide_name[1:-3]
# so that the left first column is the outcome and the rest columns are areas

print("Now running using pandas c engine WITHOUT share_memory.")
print("Our developed FFT-based MI calculation:")

mi_output = mi.binary_screening_csv_parallel(csv_file,
                                             _usecols=abide_name,
                                             csv_engine="c",
                                             sample=1250000,
                                             multp=10,
                                             core_num=10,
                                             share_memory=False)
np.save(r"./ABIDE_diagnosis_MI_output", mi_output)

print("Pearson's correlation calculation:")

pearson_output = mi.Pearson_screening_csv_parallel(csv_file,
                                                   _usecols=abide_name,
                                                   csv_engine="c",
                                                   sample=1250000,
                                                   multp=10,
                                                   core_num=10,
                                                   share_memory=False)
np.save(r"./ABIDE_diagnosis_Pearson_output", pearson_output)
