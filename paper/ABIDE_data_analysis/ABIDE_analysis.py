#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from dask import dataframe as dd
import matplotlib.pyplot as plt
import fastHDMI as mi


# # Calculate MI for ABIDE data (besides sex and age as they will be included regardless)

# In[ ]:


csv_file = r"/home/kyang/projects/def-cgreenwo/abide_data/abide_fs60_vout_fwhm0_lh_SubjectIDFormatted_N1050_nonzero_withSEX.csv"
abide = pd.read_csv(csv_file, encoding='unicode_escape', engine="c")
abide = dd.read_csv(csv_file, sample=1250000)

_abide_name = abide.columns.tolist()  #[1:]
_abide_name = list(abide.columns)  #[1:]

print(_abide_name)

abide_name = [_abide_name[-1]] + _abide_name[1:-3]
# so that the left first column is the outcome and the rest columns are areas

mi_output = mi.binary_filter_csv_parallel(csv_file,
                                          _usecols=abide_name,
                                          csv_engine="c")
np.save(r"/home/kyang/ABIDE_MI_output", mi_output)

pearson_output = mi.Pearson_filter_csv_parallel(csv_file,
                                                _usecols=abide_name,
                                                csv_engine="c")
np.save(r"/home/kyang/ABIDE_Pearson_output", pearson_output)


# # Plots

# In[20]:


abide_mi = np.load(r"./ABIDE_MI_output.npy")
print(np.sum(abide_mi > .05))
plt.hist(np.log(abide_mi), 100)


# In[ ]:




