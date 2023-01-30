# fastHDMI -- fast High-Dimensional Mutual Information estimation
## Kai Yang
## <kai.yang2@mail.mcgill.ca>

This packages uses mutual information and accelerated gradient method to screen for important variables from (potentially very) high-dimensional large datasets. As a feature, it can be applied on large `.csv` data in parallel in a memory-efficient manner and use FFT for KDE to estimate the mutual information extremely fast. A tqdm progress bar is now added to be more useful on cloud computing platforms. The corresponding paper by Yang et al. is coming soon...

The available functions are:
- `continuous_filter_plink` caculates the mutual information between a continuous outcome and a bialletic SNP using FFT. Missing data in the input variables is acceptable and will be removed per bivariate calculation. The arguments are:
  * `bed_file`, `bim_file`, `fam_file` are the location of the plink1 files;
  * `outcome`, `outcome_iid` are the outcome values and the iids for the outcome. For genetic data, it is usual that the order of SNP iid and the outcome iid don't match. While SNP iid can be obtained from the plink1 files, outcome iid here is to be declared separately. `outcome_iid` should be a list of strings or a one-dimensional numpy string array.
  * `N=500` is the default values for grid size for FFT.

- `binary_filter_plink` works similarly. 

- `continuous_filter_plink_parallel` and `binary_filter_plink_parallel` are the multiprocessing version of the above two functions, with `core_num` can be used to declare the number of cores to be used for multiprocessing.

- `MI_continuous_continuous` and `MI_binary_continuous` are to calculate mutual information between two continuous variables and binary and continuous variables, respectively. `MI_binary_012` and `MI_012_012` are `jit` complied functions -- the later can be used for clumping for very large genetic datasets.

- `binary_filter_csv`, `continuous_filter_csv`, `binary_filter_csv_parallel`, and `continuous_filter_csv_parallel` are to work on large CSV files directly in a memory efficient manner. **Note that it is assumed the left first column should be the outcome;** if not, use `_usecols` to set the first element to be the outcome column label.
  * `_usecols` is a list of column labels to be used, **the first element should be the outcome. Returned mutual information calculation results match `_usecols`.**
  * `Pearson_filter_csv_parallel` calculate Pearson's correlation between only the outcome and the covariates in similiar manner -- since `pandas.DataFrame.corr` calculate pairwise Pearson's correlation for the entire dataframe.
  * `csv_engine` can use `dask` for low memory situations, or `pandas`'s `read_csv` `engine`s, or `fastparquet` engine for a created `parquet` file for faster speed. If `fastparquet` is chosen, declare `parquet_file` as the filepath to the parquet file; if `dask` is chosen to read very large CSV, it might need to specify a larger [`sample`](https://docs.dask.org/en/stable/generated/dask.dataframe.read_csv.html).


- `UAG_LM_SCAD_MCP`, `UAG_logistic_SCAD_MCP`: these functions find a local minizer for the SCAD/MCP penalized linear models/logistic models. The arguments are:
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

- `UAG_LM_SCAD_MCP_strongrule`, `UAG_logistic_SCAD_MCP_strongrule` work just like `UAG_LM_SCAD_MCP`, `UAG_logistic_SCAD_MCP` -- except they use strong rule to filter out many covariates before carrying out the optimization step. Same for `solution_path_LM_strongrule` and `solution_path_logistic_strongrule`. Strong rule increases the computational speed dramatically.

- `SNP_UAG_LM_SCAD_MCP` and `SNP_UAG_logistic_SCAD_MCP` work similar to `UAG_LM_SCAD_MCP` and `UAG_logistic_SCAD_MCP`; and `SNP_solution_path_LM` and `SNP_solution_path_logistic` work similar to `solution_path_LM`, `solution_path_logistic` -- except that it takes plink1 files so it will be more memory-efficient. Since PCA adjustment is usually used to adjust for population structure, PCA can be given for `pca` as a 2-d array -- each column should be one principal component. The pca version is `SNP_UAG_LM_SCAD_MCP_PCA` and `SNP_UAG_logistic_SCAD_MCP_PCA`.
