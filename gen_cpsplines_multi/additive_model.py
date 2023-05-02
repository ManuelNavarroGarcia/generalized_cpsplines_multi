from typing import Dict, Iterable, List, Optional, Tuple, Union

import mosek.fusion
import numpy as np
import pandas as pd
from cpsplines.mosek_functions.interval_constraints import IntConstraints
from cpsplines.psplines.bspline_basis import BsplineBasis
from cpsplines.psplines.penalty_matrix import PenaltyMatrix
from cpsplines.utils.fast_kron import matrix_by_transpose
from cpsplines.utils.weighted_b import get_idx_fitting_region
from scipy.linalg import block_diag
from scipy.optimize import minimize
from statsmodels.genmod.families.family import Binomial


class NumericalError(Exception):
    pass


def quadratic_term(
    sp: Iterable[Union[int, float]],
    obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the matrices involved in the quadratic term of the objective
    function.

    Parameters
    ----------
    sp : Iterable[Union[int, float]]
        An array containing the smoothing parameters.
    obj_matrices : Dict[str, Union[np.ndarray, Iterable[np.ndarray]]]
        A dictionary containing the necessary arrays (the basis matrices, the
        penalty matrices and the response variable sample) used to compute the
        quadratic terms in the objective function.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The first element refers to the quadratic term of the design matrices,
        while the second element corresponds to the penalization term.
    """

    # Get the estimators of the response variable, and the weigths
    mu = Binomial().starting_mu(obj_matrices["y"])
    W = Binomial().weights(mu)
    # Compute the quadratic terms
    bases_term = block_diag(*[B.T @ (W * B.T).T for B in obj_matrices["B"][1:]])
    penalty_term = block_diag(
        *[np.multiply(s, P) for P, s in zip(obj_matrices["D_mul"], sp)]
    )
    return (bases_term, penalty_term)


def fit_irls(
    obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    penalty_term: np.ndarray,
    tol: Union[int, float] = 1e-8,
    maxiter: int = 100,
    verbose: bool = False,
) -> np.ndarray:
    """Run the IRLS algorithm for the Binomial family.

    Parameters
    ----------
    obj_matrices : Dict[str, Union[np.ndarray, Iterable[np.ndarray]]]
        A dictionary containing the necessary arrays (the basis matrices, the
        penalty matrices and the response variable sample) used to compute the
        quadratic terms in the objective function.
    penalty_term : np.ndarray
        The penalization term.
    tol : Union[int, float], optional
        The tolerance of the algorithm, by default 1e-8.
    maxiter : int, optional
        The maximum number of iterations, by default 100.
    verbose : bool, optional
        If True, messages will be displayed during the execution, by default
        False.

    Returns
    -------
    np.ndarray
        The optimal value for the target variable.
    """
    # Obtain an initial value of the fitting coefficients
    theta_old = np.zeros(np.sum([mat.shape[1] for mat in obj_matrices["B"][1:]]))
    # Use this initial value to estimate initial values for `mu` (mean of the
    # exponential family) and `eta` (transformed mu through the link function)
    mu = Binomial().starting_mu(obj_matrices["y"])
    eta = Binomial().predict(mu)

    for iter in range(maxiter):
        # Get the weights and the modified dependent variable
        W = Binomial().weights(mu)
        Z = eta + Binomial().link.deriv(mu) * (obj_matrices["y"] - mu)

        # With this modified dependent variable, update the coefficients
        bases_term = block_diag(*[B.T @ (W * B.T).T for B in obj_matrices["B"][1:]])

        T = np.multiply(W, Z)
        F = block_diag(
            *[np.expand_dims(np.dot(B.T, T), axis=1) for B in obj_matrices["B"][1:]]
        )
        theta = np.linalg.solve(bases_term + penalty_term, F).sum(axis=1)

        eta = np.dot(np.concatenate((obj_matrices["B"][1:]), axis=1), theta)
        mu = Binomial().fitted(eta)
        # Check convergence
        if np.linalg.norm(theta - theta_old) < tol:
            if verbose:
                print(f"Algorithm has converged after {iter} iterations.")
            break
        # Update the initial value of the coefficients
        theta_old = theta.copy()
    return mu


def GCV(
    sp: Iterable[Union[int, float]],
    obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
) -> float:
    """
    Computes the Generalized Cross Validation (Golub et al., 1979).

    Parameters
    ----------
    sp : Iterable[Union[int, float]]
        An array containing the smoothing parameters.
    obj_matrices : Dict[str, Union[np.ndarray, Iterable[np.ndarray]]]
        A dictionary containing the necessary arrays (the basis matrices, the
        penalty matrices and the response variable sample) used to compute the
        quadratic terms in the objective function.

    References
    ----------
    ... [1] GOLUB, G. H., HEATH, M., and WAHBA, G. (1979). Generalized
    cross-validation as a method for choosing a good ridge parameter.
    Technometrics, 21(2), 215-223.

    Returns
    -------
    float
        The GCV value.
    """

    bases_term, penalty_term = quadratic_term(sp=sp, obj_matrices=obj_matrices)
    y_hat = fit_irls(obj_matrices=obj_matrices, penalty_term=penalty_term)
    # Return the GCV value, which is n * RSS / (n - tr(H))**2, where RSS is the
    # residual sum of squares, n is the product of the dimensions of y and H is
    # the hat matrix of the unconstrained problem
    return (
        len(obj_matrices["y"]) * np.square(np.linalg.norm((obj_matrices["y"] - y_hat)))
    ) / np.square(
        len(obj_matrices["y"])
        - np.trace(np.linalg.solve(bases_term + penalty_term, bases_term))
    )


class AdditiveModel:
    def __init__(
        self,
        deg: Iterable[int] = (3,),
        ord_d: Iterable[int] = (2,),
        n_int: Iterable[int] = (10,),
        constrained: bool = False,
        x_range: Optional[Dict[str, Tuple[Union[int, float]]]] = None,
    ):
        self.deg = deg
        self.ord_d = ord_d
        self.n_int = n_int
        self.constrained = constrained
        self.x_range = x_range

    def _get_bspline_bases(self, x: Iterable[np.ndarray]) -> List[BsplineBasis]:
        """
        Construct the B-spline bases on each axis.

        Parameters
        ----------
        x : Iterable[np.ndarray]
            The covariates samples.

        Returns
        -------
        List[BsplineBasis]
            The list of B-spline bases on each axis.
        """

        bspline_bases = []
        if self.x_range is None:
            self.x_range = {}
        for deg, xsample, n_int, name in zip(
            self.deg, x, self.n_int, self.feature_names
        ):
            # Get the maximum and minimum of the fitting regions
            x_min, x_max = np.min(xsample), np.max(xsample)
            prediction_dict = {}
            if name in self.x_range:
                # If the values in `x_range` are outside the fitting region,
                # include them in the `prediction` argument of the BsplineBasis
                pred_min, pred_max = min(self.x_range[name]), max(self.x_range[name])
                if pred_max > x_max:
                    prediction_dict["forward"] = pred_max
                if pred_min < x_min:
                    prediction_dict["backwards"] = pred_min
            bsp = BsplineBasis(
                deg=deg, xsample=xsample, n_int=n_int, prediction=prediction_dict
            )
            # Generate the design matrix of the B-spline basis
            bsp.get_matrix_B()
            bsp.get_matrices_S()
            bspline_bases.append(bsp)
        return bspline_bases

    def _get_obj_func_arrays(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Get the arrays involved in the objective function of the optimization
        problem.

        Parameters
        ----------
        y : np.ndarray
            The response variable sample.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary containing the matrices involved in the construction of
            the objective function:
            - "B": the design matrices (with the intercept).
            - "D": the difference matrices.
            - "D_mul": the penalty matrices (without multiplying by the
            smoothing parameter).
            - "y": the response variable sample.
        """

        obj_matrices = {}
        obj_matrices["B"] = [np.ones((len(y), 1))]
        obj_matrices["D"] = []
        obj_matrices["D_mul"] = []
        indexes_fit = get_idx_fitting_region(self.bspline_bases)
        for bsp, ord_d, idx in zip(self.bspline_bases, self.ord_d, indexes_fit):
            B = bsp.matrixB
            obj_matrices["B"].append(B[idx])
            penaltymat = PenaltyMatrix(bspline=bsp)
            P = penaltymat.get_penalty_matrix(**{"ord_d": ord_d})
            obj_matrices["D"].append(penaltymat.matrixD)
            obj_matrices["D_mul"].append(P)

        obj_matrices["y"] = y.copy()
        return obj_matrices

    def _initialize_model(
        self,
        obj_matrices: Union[np.ndarray, Iterable[np.ndarray]],
    ) -> mosek.fusion.Model:
        """
        Construct the optimization model.
        Parameters
        ----------
        obj_matrices : Dict[str, Union[np.ndarray, Iterable[np.ndarray]]
            A dictionary containing the necessary arrays (the basis matrices, the
            penalty matrices and the response variable sample) used to compute the
            quadratic terms in the objective function.

        Returns
        -------
        mosek.fusion.Model
            The MOSEK optimization model.
        """
        M = mosek.fusion.Model()
        # Define the regression coefficients decision variables
        intercept = M.variable("theta_0", 1, mosek.fusion.Domain.unbounded())
        dict_theta = {}
        dict_theta["theta_0"] = intercept
        for i, bsp in enumerate(self.bspline_bases):
            theta = M.variable(
                f"theta_{i + 1}", bsp.matrixB.shape[1], mosek.fusion.Domain.unbounded()
            )
            dict_theta[f"theta_{i + 1}"] = theta

        # Define auxiliary variables
        t_D = M.variable(
            "t_D", len(self.bspline_bases), mosek.fusion.Domain.greaterThan(0.0)
        )

        t = M.variable(
            "t", len(obj_matrices["y"]), mosek.fusion.Domain.greaterThan(0.0)
        )
        u = M.variable(
            "u", len(obj_matrices["y"]), mosek.fusion.Domain.greaterThan(0.0)
        )
        v = M.variable(
            "v", len(obj_matrices["y"]), mosek.fusion.Domain.greaterThan(0.0)
        )
        sp = [M.parameter(f"sp_{i}", 1) for i, _ in enumerate(self.deg)]

        coef = [
            mosek.fusion.Expr.mul(B, dict_theta[f"theta_{i}"])
            for i, B in enumerate(obj_matrices["B"])
        ]
        # Define the contribution of the penalty in the objective function
        for g, D in enumerate(obj_matrices["D"]):
            M.constraint(
                f"rot_cone_D_{g}",
                mosek.fusion.Expr.vstack(
                    t_D.slice(g, g + 1),
                    1 / 2,
                    mosek.fusion.Expr.mul(
                        mosek.fusion.Matrix.sparse(D), dict_theta[f"theta_{g + 1}"]
                    ),
                ),
                mosek.fusion.Domain.inRotatedQCone(),
            )

        # Auxiliary constraints for the Binomial likelihood
        M.constraint(
            mosek.fusion.Expr.hstack(
                u,
                mosek.fusion.Expr.constTerm(len(obj_matrices["y"]), 1.0),
                mosek.fusion.Expr.sub(mosek.fusion.Expr.add(coef), t),
            ),
            mosek.fusion.Domain.inPExpCone(),
        )

        M.constraint(
            mosek.fusion.Expr.hstack(
                v,
                mosek.fusion.Expr.constTerm(len(obj_matrices["y"]), 1.0),
                mosek.fusion.Expr.mul(-1, t),
            ),
            mosek.fusion.Domain.inPExpCone(),
        )

        M.constraint(mosek.fusion.Expr.add(u, v), mosek.fusion.Domain.lessThan(1.0))
        for c in coef[1:]:
            M.constraint(mosek.fusion.Expr.sum(c), mosek.fusion.Domain.equalsTo(0.0))

        # Define the objective function
        obj = mosek.fusion.Expr.add(
            [mosek.fusion.Expr.dot(s, t_D.slice(i, i + 1)) for i, s in enumerate(sp)]
        )
        obj = mosek.fusion.Expr.sub(
            mosek.fusion.Expr.add(
                mosek.fusion.Expr.sum(t),
                obj,
            ),
            mosek.fusion.Expr.add(
                [
                    mosek.fusion.Expr.dot(obj_matrices["y"].astype(float), c)
                    for c in coef
                ]
            ),
        )
        M.objective("obj", mosek.fusion.ObjectiveSense.Minimize, obj)

        return M

    def _non_decreasing_constraints(
        self, model: mosek.fusion.Model
    ) -> mosek.fusion.Model:
        """Define the non-decreasing constraints and include them into the
        optimization problem.

        Parameters
        ----------
        model : mosek.fusion.Model
            The MOSEK optimization model.

        Returns
        -------
        mosek.fusion.Model
            The MOSEK optimization model.
        """
        for i, bsp in enumerate(self.bspline_bases):
            S = bsp.matrices_S
            int_cons = IntConstraints(
                bspline=[bsp], var_name=0, derivative=1, constraints={"+": 0.0}
            )
            int_cons.interval_cons(
                model=model,
                matrices_S={0: S},
                var_dict={"theta": model.getVariable(f"theta_{i+1}")},
            )
        return model

    def _get_sp_optimizer(
        self,
        obj_matrices: Dict[str, Union[np.ndarray, Iterable[np.ndarray]]],
    ) -> Tuple[Union[int, float]]:
        """Get the optimal smoothing parameters.

        Parameters
        ----------
        obj_matrices : Dict[str, Union[np.ndarray, Iterable[np.ndarray]]]
            A dictionary containing the necessary arrays (the basis matrices, the
            penalty matrices and the response variable sample) used to compute the
            quadratic terms in the objective function.

        Returns
        -------
        Tuple[Union[int, float]]
            A tuple containing the optimal smoothing parameters according to the
            GCV criterion.
        """
        # Get the best set of smoothing parameters
        best_sp = minimize(
            GCV,
            x0=np.ones((len(obj_matrices["D"]), 1)),
            args=(obj_matrices,),
            method="L-BFGS-B",
            bounds=[(1e-10, 1e16) for _ in range(len(self.deg))],
        ).x
        return best_sp

    def fit(self, data: pd.DataFrame, y_col: str):
        """Fit the data using non-decreasing additive P-splines

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing the covariates and the response variable
            samples.
        y_col : str
            The name of the response variable.

        Raises
        ------
        NumericalError
            Raises if MOSEK could not find a solution due to numerical errors.
        """
        # Define the regressor samples and the target variable sample
        self.feature_names = data.drop(columns=y_col).columns

        x = [data[col].values for col in data if not col.startswith(y_col)]
        y = data[y_col].values

        # Define the B-spline basis
        self.bspline_bases = self._get_bspline_bases(x=x)
        obj_matrices = self._get_obj_func_arrays(y=y)

        obj_matrices["B_mul"] = list(map(matrix_by_transpose, obj_matrices["B"][1:]))

        # Initialize the model
        M = self._initialize_model(obj_matrices=obj_matrices)

        if self.constrained:
            M = self._non_decreasing_constraints(model=M)

        # Get the optimal smoothing parameter and set them
        self.best_sp = self._get_sp_optimizer(obj_matrices=obj_matrices)
        for i, sp in enumerate(self.best_sp):
            M.getParameter(f"sp_{i}").setValue(sp)
        # Solve the optimization model
        try:
            M.solve()
            self.sol = [
                M.getVariable(f"theta_{i}").level()
                for i in range(len(obj_matrices["B"]))
            ]

        except mosek.fusion.SolutionError as e:
            raise NumericalError(f"The original error was {e}")

    def predict(self, data: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        """Generates output predictions for the input samples.

        Parameters
        ----------
        data : Union[pd.Series, pd.DataFrame]
            The input data where the predictions are to be computed.

        Returns
        -------
        np.ndarray
            Numpy array(s) of predictions.

        Raises
        ------
        ValueError
            If some of the coordinates are outside the definition range of the
            B-spline bases.
        """
        # If no data is inputted, return an empty array
        if data.empty:
            return np.array([])

        # Data must be in DataFrame for so the transpose can be performed in the
        # next steps
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)

        x = [row for row in data.values.T]
        x_min = np.array([np.min(v) for v in x])
        x_max = np.array([np.max(v) for v in x])
        bsp_min = np.array([bsp.knots[bsp.deg] for bsp in self.bspline_bases])
        bsp_max = np.array([bsp.knots[-bsp.deg] for bsp in self.bspline_bases])
        # If some coordinates are outside the range where the B-spline bases
        # were defined, the problem must be fitted again
        if (x_min < bsp_min).sum() > 0:
            raise ValueError(
                "Some of the coordinates are outside the definition range of "
                "the B-spline bases."
            )
        if (x_max > bsp_max).sum() > 0:
            raise ValueError(
                "Some of the coordinates are outside the definition range of "
                "the B-spline bases."
            )
        # Compute the basis matrix at the coordinates to be predicted
        B_predict = [np.ones((len(data), 1))] + [
            bsp.bspline_basis(x=x[i]) for i, bsp in enumerate(self.bspline_bases)
        ]
        return Binomial().fitted(
            np.sum([np.dot(B, sol) for B, sol in zip(B_predict, self.sol)], axis=0)
        )
