# generalized_cpsplines_multi

`generalized_cpsplines_multi` is a GitHub repository containing all the figures
and simulations results shown in the paper

All the simulation studies carried out in this work use the routines implemented
in [cpsplines](https://github.com/ManuelNavarroGarcia/cpsplines), which requires
a [MOSEK](https://www.mosek.com) license to solve the optimization problems.

## Project structure

The current version of the project is structured as follows:

* **generalized_cpsplines_multi**: the main directory of the project, which
  consist of:
  * **additive_models.py**: contains the code to fit constrained additive
    P-splines.
  * **figures.ipynb**: A Jupyter notebook containing the code used to generate
    the figures and the tables of the paper.
  * **hschool.py**: the code used to carry out the case-study with the `hschool`
    data set.
  * **pima.py**: the code used to carry out the case-study with the `pima`
    data set.
  * **R_scripts.R**: contains the R code of the packages `scam` and `cgam` used
    during the real case-studies.
  * **R_utils.R**: an R script with some useful functions.
  * **simulations.py**: the code used to carry out the simulation study.
* **data**: a folder containing CSV and parquet files with simulated and real
  data sets, and the results of the applications.
* **img**: a folder containing the figures shown in the paper.

## Package dependencies

`generalized_cpsplines_multi` mainly depends on the following packages:

* [cpsplines](https://pypi.org/project/cpsplines/).
* [pyarrow](https://pypi.org/project/pyarrow/)
* [rpy2](https://pypi.org/project/rpy2/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [typer](https://pypi.org/project/typer/)

## Contact Information and Citation

If you have encountered any problem or doubt while using
`generalized_cpsplines_multi`, please feel free to let me know by sending me an
email:

* Name: Manuel Navarro García (he/his)
* Email: manuelnavarrogithub@gmail.com

If you find `generalized_cpsplines_multi` or `cpsplines` useful, please cite it
in your publications.
