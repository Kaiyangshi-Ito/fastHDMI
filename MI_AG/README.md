# MI_AG

Use mutual information and accelerated gradient method to filter out and optimize nonconvex sparse learning problems on large genetic data based on bed/bim/fam. The corresponding paper is coming soon...

The available functions are:
- `continuous_filter` caculates the mutual information between a continuous outcome and a bialletic SNP using FFT. Missing data is acceptable and will be removed. The arguments are:
  * `bed_file`, `bim_file`, `fam_file` are the location of the plink1 files;
  * `outcome`, `outcome_iid` are the outcome values and the iids for the outcome. For genetic data, it is usual that the order of SNP iid and the outcome iid don't match. While SNP iid can be obtained from the plink1 files, outcome iid here is to be declared separately. `outcome_iid` should be a list of strings or a one-dimensional numpy string array.
  * `a_min`, `a_max` are the minimum and maximum of the continous outcome used to evaluate the support; `N=500` is the default values for grid size for FFT.

- `binary_filter` works similarly, execpt that `a_min=0, a_max=1` obivously.

- `continuous_filter_parallel` and `binary_filter_parallel` are the multiprocessing version of the above two functions, with `chunck_size=60000` can be used to declare the chunk size.


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

- `SNP_UAG_LM_SCAD_MCP` and `SNP_UAG_logistic_SCAD_MCP` work similar to `UAG_LM_SCAD_MCP` and `UAG_logistic_SCAD_MCP`; and `SNP_solution_path_LM` and `SNP_solution_path_logistic` work similar to `solution_path_LM`, `solution_path_logistic` -- except that it takes plink1 files so it will be more memory-efficient.
