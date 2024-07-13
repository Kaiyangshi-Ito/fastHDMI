# fastHDMI -- fast High-Dimensional Mutual Information estimation
## Kai Yang
## <kai.yang2 "at" mail.mcgill.ca>
## [GPG Public key Fingerprint: B9F863A56220DBD56B91C3E835022A1A5941D810](https://keys.openpgp.org/vks/v1/by-fingerprint/B9F863A56220DBD56B91C3E835022A1A5941D810)

**This repository contains codes and results for my package `fastHDMI` and my paper *{\tt fastHDMI}: Fast Mutual Information Estimation for High-Dimensional Data***

The manual for the package `fastHDMI` can be found [here](/fastHDMI/README.md). The package is published on PyPI [here](https://pypi.org/project/fastHDMI/).

The results on [(pre-processed) ABIDE data](http://preprocessed-connectomes-project.org/abide/) is summarized [in this Jupyter notebook](/paper/ABIDE_data_analysis/ABIDE_analysis.ipynb) -- running the Jupyter notebook will generate the Python and bash scripts to use the `fastHDMI` package to analyze the data. The scripts are set to run on the server, in my case, Compute Canada, and the running the Jupyter notebook again with the returned data files (i.e., `.npy`s) will yield the plots and other results used in the paper -- the output plots are in pdf format.

the `seff-[jobID].out` files are the outputs from `seff [jobsID]` command -- they briefly describes the computational resources used for the job. More on Compute Canada running jobs can be found [here](https://docs.alliancecan.ca/wiki/Running_jobs).
