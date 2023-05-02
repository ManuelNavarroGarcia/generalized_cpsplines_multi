import itertools
import logging
import time
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet
import rpy2.robjects as robjects
import typer
from cpsplines.fittings.fit_cpsplines import CPsplines, NumericalError
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO
)


def simulated_example(
    x_1: Tuple[Union[int, float], Union[int, float]],
    x_2: Tuple[Union[int, float], Union[int, float]],
    n: int,
    grid: bool = True,
) -> pd.DataFrame:
    """Given two intervals `x_1` and `x_2`, generates a sample of points in the
    generated rectangle with size `n * n` and evaluates them at the surface in
    the Example 4.1 in [1].

    Parameters
    ----------
    x_1 : Tuple[Union[int, float], Union[int, float]]
        The interval in the X_1-axis.
    x_2 : Tuple[Union[int, float], Union[int, float]]
        The interval in the X_2-axis.
    n : int
        The square root of the sample size.
    grid : bool, optional
        If True, `n` points will be drawn from uniform distributions U(`x_1`)
        and U(`x_2`) and then a grid data structure is built combining these
        positions. Otherwise, `n` * `n` samples are directly drawn from a
        uniform distribution over the rectangle `x_1` x `x_ 2`. By default True.

    Returns
    -------
    pd.DataFrame
        A DataFrame with three columns and `n` * `n` rows:
        - "x_1": The X_1-coordinates of the points in the sample.
        - "x_2": The X_2-coordinates of the points in the sample.
        - "y": The evaluations of the points (x_1, x_2) at the surface.

    References
    ----------
    ... [1] DETTE, H. and SCHEDER, R. (2006). Strictly monotone and smooth
    nonparametric regression for two or more variables. Canadian Journal of
    Statistics, 34, 535-561. https://doi.org/10.1002/cjs.5550340401
    """
    if grid:
        x_1 = np.repeat(np.sort(np.random.uniform(*x_1, size=n)), n)
        x_2 = np.tile(np.sort(np.random.uniform(*x_2, size=n)), n)
    else:
        x_1 = np.random.uniform(*x_1, size=n * n)
        x_2 = np.random.uniform(*x_2, size=n * n)
    # Check [1] for the expression fo the surface
    y = (
        (x_2 + np.sin(6 * np.pi * x_2) / (6 * np.pi))
        * (1 + np.power(2 * x_1 - 1, 3))
        / 2
    )
    return pd.DataFrame({"x1": x_1, "x2": x_2, "y": y})


def simulated_example_results(
    n_iter: int,
    n: int,
    k: int,
    x_1: Tuple[Union[int, float], Union[int, float]],
    x_2: Tuple[Union[int, float], Union[int, float]],
    grid: bool,
    sigma: Union[int, float],
    graph: Optional[Callable] = None,
    r_script: str = "R_scripts.R",
    method_name: str = "simulated_example_results_R",
) -> pd.DataFrame:
    """Performs a simulation study in shape-constrained regression, comparing
    the packages cpsplines, cgam and scam, together with the unconstrained
    P-splines.

    Parameters
    ----------
    n_iter : int
        The number of iterations.
    n : int
        The square root of the sample size.
    k : int
        The number of inner knots to be used in the construction of the basis.
    x_1 : Tuple[Union[int, float], Union[int, float]]
        The interval in the X_1-axis.
    x_2 : Tuple[Union[int, float], Union[int, float]]
        The interval in the X_2-axis.
    grid : bool, optional
        If True, `n` points will be drawn from uniform distributions U(`x_1`)
        and U(`x_2`) and then a grid data structure is built combining these
        positions. Otherwise, `n` * `n` samples are directly drawn from a
        uniform distribution over the rectangle `x_1` x `x_ 2`. By default True.
    sigma : Union[int, float]
        The standard deviation of the noise error term.
    graph : Optional[Callable], optional
        The theoretical surface in which the constraints are to be imposed. If
        None, the surface in the Example 4.1 in [1] is used. By default None.
    r_script : str, optional
        The path with the R code to execute the simulations in the packages cgam
        and scam, by default "simulated_example_R.R".
    method_name : str, optional
        The name of the function in `r_script` used to execute the simulations,
        by default "simulated_example_results_R".

    Returns
    -------
    pd.DataFrame
        The results of the simulations. The DataFrame has `n_iter` rows and 20
        columns. For each method, the following metrics are stored:
        - MAE: The Mean Absolute Error between the estimated surface and the
        observed data.
        - Theo_MAE: The Mean Absolute Error between the estimated surface and
        the theoretical surface.
        - MSE: The Mean Squared Error between the estimated surface and the
        observed data.
        - Theo_MAE: The Mean Squared Error between the estimated surface and the
        theoretical surface.
        - Times: The execution time.

    References
    ----------
    ... [1] DETTE, H. and SCHEDER, R. (2006). Strictly monotone and smooth
    nonparametric regression for two or more variables. Canadian Journal of
    Statistics, 34, 535-561. https://doi.org/10.1002/cjs.5550340401
    """

    # Define the columns of the output DataFrame with the results
    metrics = ["MAE", "Theo_MAE", "MSE", "Theo_MSE", "Times"]
    methods = ["Unconstrained", "cpsplines", "cgam", "scam"]
    out_columns = [
        "_".join(col[::-1]) for col in itertools.product(*(methods, metrics))
    ]

    # Define the theoretical surface
    if graph is None:
        graph = simulated_example
    df = graph(x_1=x_1, x_2=x_2, n=n, grid=grid)

    # Initialize the output DataFrame
    results = np.zeros((n_iter, len(out_columns)))

    for w in range(n_iter):
        np.random.seed(w)
        logging.info(f"Iteration: {w+1}")
        # Add noise to the theoretical curve
        df["y_error"] = df["y"] + np.random.normal(0, sigma, len(df))
        try:
            # Fit the data with unconstrained P-splines and update the results
            start = time.time()
            unconstrained_fit = CPsplines(
                deg=(3, 3),
                ord_d=(2, 2),
                n_int=(k, k),
                sp_method="optimizer",
                sp_args={"method": "L-BFGS-B"},
            )
            _ = unconstrained_fit.fit(data=df.drop(columns=["y"]), y_col="y_error")
            y_unconstrained = unconstrained_fit.predict(data=df[["x1", "x2"]])
            end = time.time()

            results[w, : len(metrics)] = np.array(
                [
                    mean_absolute_error(y_unconstrained, df["y_error"]),
                    mean_absolute_error(y_unconstrained, df["y"]),
                    mean_squared_error(y_unconstrained, df["y_error"]),
                    mean_squared_error(y_unconstrained, df["y"]),
                    end - start,
                ]
            )
            # Fit the data with double non-decreasing P-splines and update the
            # results
            start = time.time()
            cpsplines_fit = CPsplines(
                deg=(3, 3),
                ord_d=(2, 2),
                n_int=(k, k),
                sp_method="optimizer",
                sp_args={"method": "L-BFGS-B"},
                int_constraints={0: {1: {"+": 0}}, 1: {1: {"+": 0}}},
            )
            _ = cpsplines_fit.fit(data=df.drop(columns=["y"]), y_col="y_error")
            y_cpsplines = cpsplines_fit.predict(data=df[["x1", "x2"]])
            end = time.time()
            results[w, len(metrics) : 2 * len(metrics)] = np.array(
                [
                    mean_absolute_error(y_cpsplines, df["y_error"]),
                    mean_absolute_error(y_cpsplines, df["y"]),
                    mean_squared_error(y_cpsplines, df["y_error"]),
                    mean_squared_error(y_cpsplines, df["y"]),
                    end - start,
                ]
            )
            # Defining the R script and loading the instance in Python
            r = robjects.r
            r["source"](r_script)
            # Loading the function we have defined in R
            scam_cgam_eval_r = robjects.globalenv[method_name]
            # converting it into r object for passing into r function
            with localconverter(robjects.default_converter + pandas2ri.converter):
                df_r = robjects.conversion.py2rpy(df)
            # Invoking the R function and getting the result
            results[w, 2 * len(metrics) :] = np.asarray(scam_cgam_eval_r(df_r, k))
        except RRuntimeError or NumericalError:
            logging.warning(f"Iteration {w+1} has not reached to the expected solution")
            results[w, :] = np.array([np.nan] * len(out_columns))
    results_df = pd.DataFrame(data=results, columns=out_columns)
    return results_df.reindex(sorted(results_df.columns), axis=1).dropna()


def main(
    r_script: str = "gen_cpsplines_multi/R_scripts.R",
    fout: str = "data/simulated_example_results.parquet",
):
    sigmas = (0.05, 0.10, 0.25, 0.50)
    gridded = (True, False)

    L = []
    for grid in gridded:
        ls = []
        for sigma in sigmas:
            df = simulated_example_results(
                n_iter=100,
                n=30,
                k=30,
                x_1=(0, 1),
                x_2=(0, 1),
                grid=grid,
                sigma=sigma,
                r_script=r_script,
            )
            ls.append(df)
        # Concatenate the DataFrame for common data structure
        df_ = pd.concat([pd.concat([s], axis=1).T for s in ls], axis=0, keys=sigmas)
        L.append(df_)
    # Concatenate DataFrames with gridded and scattered data
    out = pd.concat([pd.concat([s], axis=1).T for s in L], axis=1, keys=gridded)
    table = pyarrow.Table.from_pandas(out.T)
    pyarrow.parquet.write_table(table, fout)


if __name__ == "__main__":
    typer.run(main)
