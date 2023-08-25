# fastHDMI -- fast High-Dimensional Mutual Information estimation
## Kai Yang
## <kai.yang2 "at" mail.mcgill.ca>
## License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)
## [GPG Public key Fingerprint: CC02CF153594774CF956691492B2600D18170329](https://keys.openpgp.org/vks/v1/by-fingerprint/CC02CF153594774CF956691492B2600D18170329)

This packages uses FFT-based mutual information screening and accelerated gradient method for important variables from (potentially very) high-dimensional large datasets. **version `1.23.23` is a version with only the README file updated to illustrate the functions more clearly**

Consider the sizes of the datafiles, the most commonly-used functions are the functions run in parallel -- all functions running in parallel will has `_parallel` suffix; and they all have arguments: 
- `core_num`: number of CPU cores used for multiprocessing; the default option is to use all the cores available, considering the job is most likely running on a server instead of a PC 
- `multp`: job multiplier, the job to be run in parallel will be first divided into `core_num * multp` sub-jobs -- as equal as possible, then at each time, one core will take one subjob.
- `verbose`: how verbal the function will be, with `0` being least verbal and increases wrt. the number decalred here

The function implementing our propsoed FFT-based mutual information estimation will have the following arguments:
- `N`: the grid size for 1-D FFT; with `N=500` as the default value
- `a_N`, `a_N`: similar to above, the grid size for 2-D FFT; with `300` as the default values 
- `kernel` and `bw` specify the kernel and bandwidth used for KDE
- `norm` is the norm used for KDE -- this option only takes effects for 2-D KDE

The screening functions and their arguments: 

- For `plink` files:
* arguments `bed_file`, `bim_file`, `fam_file` are the location of the plink files;
* arguments `outcome`, `outcome_iid` are the outcome values and the iids for the outcome. For genetic data, it is usual that the order of SNP iid and the outcome iid don't match. While SNP iid can be obtained from the plink1 files, outcome iid here is to be declared separately. `outcome_iid` should be a list of strings or a one-dimensional numpy string array.
* `continuous_screening_plink`, `continuous_screening_plink_parallel` for screening on continuous outcomes with continuous covariates  
* `binary_screening_plink`, `binary_screening_plink_parallel` for screening on binary outcomes with continuous covariates
* `clump_plink_parallel` for clumping -- starting from the first covariate (i.e., the first column on the left of the datafile), clumping will remove all subsequent covariates with a mutual information higher than what the `clumping_threshold` declares with the one it looks at

- For `csv` files: 
* argument `_usecols` is a list of column labels to be used, **the first element should be the outcome. Returned mutual information calculation results match `_usecols`.**
* **Note that it is assumed the left first column should be the outcome;** if not, use `_usecols` to set the first element to be the outcome column label. 
* `csv_engine` can use `dask` for low memory situations, or `pandas`'s `read_csv` `engine`s, or `fastparquet` engine for a created `parquet` file for faster speed. If `fastparquet` is chosen, declare `parquet_file` as the filepath to the parquet file; if `dask` is chosen to read very large CSV, it might need to specify a larger [`sample`](https://docs.dask.org/en/stable/generated/dask.dataframe.read_csv.html).
* `binary_screening_csv`, `binary_screening_csv_parallel` for screening on binary outcomes with continuous covariates  
* `binary_skMI_screening_csv_parallel`, `continuous_skMI_screening_csv_parallel` for screening using mutual information estimation provided by `skLearn`, i.e., `sklearn.metrics.mutual_info_score`, `sklearn.feature_selection.mutual_info_classif`
* `Pearson_screening_csv_parallel` for screening using Pearson correlation 
* `continuous_screening_csv`, `continuous_screening_csv_parallel`  for screening on continuous outcomes with continuous covariates  
* `clump_continuous_csv_parallel` similar to above 

A `share_memory` option is added for multiprocess computing. As a feature, it can be applied on large `.csv` data in parallel in a memory-efficient manner and use FFT for KDE to estimate the mutual information extremely fast. A tqdm progress bar is now added to be more useful on cloud computing platforms. `verbose` option can take values of `0,1,2`, with `2` being most verbal; `1` being only show progress bar, and `0` being not verbal at all.

- For DataFrame files:
* `binary_screening_dataframe`, `binary_screening_dataframe_parallel` for screening on binary outcomes with continuous covariates  
* `binary_skMI_screening_dataframe_parallel`, `continuous_skMI_screening_dataframe_parallel` for screening using mutual information estimation provided by `skLearn`, i.e., `sklearn.metrics.mutual_info_score`, `sklearn.feature_selection.mutual_info_classif`
* `Pearson_screening_dataframe_parallel`  for screening using Pearson correlation 
* `continuous_screening_dataframe`, `continuous_screening_dataframe_parallel` for screening on continuous outcomes with continuous covariates  
* `clump_continuous_dataframe_parallel` similar to above 

- For `numpy` arrays:
* `binary_screening_array`, `binary_screening_array_parallel` for screening on binary outcomes with continuous covariates  
* `continuous_screening_array`, `continuous_screening_array_parallel` for screening on continuous outcomes with continuous covariates  
* `binary_skMI_array_parallel`, `continuous_skMI_array_parallel` for screening using mutual information estimation provided by `skLearn`, i.e., `sklearn.metrics.mutual_info_score`, `sklearn.feature_selection.mutual_info_classif`
* `continuous_Pearson_array_parallel`  for screening using Pearson correlation 




<!-- - `UAG_LM_SCAD_MCP`, `UAG_logistic_SCAD_MCP`: these functions find a local minizer for the SCAD/MCP penalized linear models/logistic models. The arguments are:
  * `design_matrix`: the design matrix input, should be a two-dimensional numpy array;
  * `outcome`: the outcome, should be one dimensional numpy array, continuous for linear model, binary for logistic model;
  * `beta_0`: starting value; optional, if not declared, it will be calculated based on the Gauss-Markov theory estimators of $\beta$;
  * `tol`: tolerance parameter; the tolerance parameter is set to be the uniform norm of two iterations;
  * `maxit`: maximum number of iteratios allowed;
  * `_lambda`: _lambda value;
  * `penalty`: could be `"SCAD"` or `"MCP"`;
  * `a=3.7`, `gamma=2`: `a` for SCAD and `gamma` for MCP; it is recommended for `a` to be set as $3.7$;
  * `L_convex`: the L-smoothness constant for the convex component, if not declared, it will be calculated by itself
  * `add_intercept_column`: boolean, should the fucntion add an intercept column?

- `solution_path_LM`, `solution_path_logistic`: calculate the solution path for linear/logistic models; the only difference from above is that `lambda_` is now a one-dimensional numpy array for the values of $\lambda$ to be used.

- `UAG_LM_SCAD_MCP_strongrule`, `UAG_logistic_SCAD_MCP_strongrule` work just like `UAG_LM_SCAD_MCP`, `UAG_logistic_SCAD_MCP` -- except they use strong rule to screening out many covariates before carrying out the optimization step. Same for `solution_path_LM_strongrule` and `solution_path_logistic_strongrule`. Strong rule increases the computational speed dramatically.

- `SNP_UAG_LM_SCAD_MCP` and `SNP_UAG_logistic_SCAD_MCP` work similar to `UAG_LM_SCAD_MCP` and `UAG_logistic_SCAD_MCP`; and `SNP_solution_path_LM` and `SNP_solution_path_logistic` work similar to `solution_path_LM`, `solution_path_logistic` -- except that it takes plink1 files so it will be more memory-efficient. Since PCA adjustment is usually used to adjust for population structure, PCA can be given for `pca` as a 2-d array -- each column should be one principal component. The pca version is `SNP_UAG_LM_SCAD_MCP_PCA` and `SNP_UAG_logistic_SCAD_MCP_PCA`. -->
