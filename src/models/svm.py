"""
robust soft-margin support vector machine
"""

# load packages
import os
from typing import List, Tuple
import numpy as np

import gurobipy as gp
from gurobipy import GRB

# allowing nonconvex constraints
gp.setParam("NonConvex", 2)


class SVMBase:
    """sklearn like interface, implementing fit and predict"""

    def __init__(
        self, gamma: float, selected_feature_list: List, verbose: bool = False
    ) -> None:
        """
        :param gamma: the penalization multiplier
        :param selected_feature_list: the index of selected perturbations
        :param verbose: True to print model update at each iterations
        """
        self.gamma = gamma
        self.selected_feature_list = selected_feature_list
        self.verbose = verbose

    def _get_params(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gamma: float,
        selected_feature_list: List,
        verbose: bool = False,
    ) -> Tuple[np.ndarray]:
        """initialize model and return"""
        raise NotImplementedError()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        training process using optimization
        :param X: the training predictors
        :parma y: the training labels
        """
        # get model and optimize
        self.beta, self.b = self._get_params(
            X, y, self.gamma, self.selected_feature_list, self.verbose
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        make prediction using testing predictors
        :param X: test predictors
        """
        raw = (X @ self.beta - self.b).flatten()  # different from lr
        predictions = ((raw > 0) * 2 - 1).astype(int)
        return predictions


class SVMRegular(SVMBase):
    """regularized version of SVM"""

    def _get_params(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gamma: float,
        selected_feature_list: List,
        verbose: bool = False,
    ) -> Tuple[np.ndarray]:
        # initialize model
        n, p = X.shape
        model = gp.Model("SVMRegular")

        if not verbose:
            model.setParam("OutputFlag", 0)

        # -------- scalar parameters -------
        # suffices to search within a unit cube, since only the direction matters
        beta = model.addVars(range(p), vtype=GRB.CONTINUOUS, lb=-10, ub=10)
        b = model.addVar(vtype=GRB.CONTINUOUS)
        xi = model.addVars(range(n), vtype=GRB.CONTINUOUS)
        beta_norm = model.addVar(vtype=GRB.CONTINUOUS)

        # -------- constraints -------
        # partially penalized l2 norm constraint
        model.addConstr(
            beta_norm * beta_norm
            == sum([beta[j] * beta[j] for j in selected_feature_list])
        )

        model.addConstrs(
            (
                (y[i] * (sum([X[i, j] * beta[j] for j in range(p)]) - b)) >= 1 - xi[i]
                for i in range(n)
            )
        )
        model.addConstrs(xi[i] >= 0 for i in range(n))

        # ------- objective --------
        model.setObjective(
            sum(xi[i] + gamma * beta_norm for i in range(n)), GRB.MINIMIZE
        )

        # optimize
        model.optimize()

        # get optimized values
        opt_beta = [beta[j].x for j in range(len(beta))]
        opt_b = b.x

        # convert to numpy
        beta = np.array(opt_beta).reshape(-1, 1)
        b = opt_b
        return beta, b


class SVMRobust(SVMBase):
    """robust version of SVM"""

    def _get_params(
        self,
        X: np.ndarray,
        y: np.ndarray,
        gamma: float,
        selected_feature_list: List,
        verbose: bool = False,
    ) -> Tuple[np.ndarray]:
        # initialize model
        n, p = X.shape
        model = gp.Model("SVMRegular")
        if not verbose:
            model.setParam("OutputFlag", 0)

        # -------- scalar parameters -------
        beta = model.addVars(
            range(p), vtype=GRB.CONTINUOUS, lb=-10, ub=10
        )  # suffices to search within a unit cube, since only the direction matters
        b = model.addVar(vtype=GRB.CONTINUOUS)
        xi = model.addVars(range(n), vtype=GRB.CONTINUOUS)
        beta_norm = model.addVar(vtype=GRB.CONTINUOUS)

        # -------- constraints -------
        # partially penalized l2 norm constraint
        model.addConstr(
            beta_norm * beta_norm
            == sum([beta[j] * beta[j] for j in selected_feature_list])
        )

        model.addConstrs(
            (
                (
                    y[i] * (sum([X[i, j] * beta[j] for j in range(p)]) - b)
                    - gamma * beta_norm
                )
                >= 1 - xi[i]
                for i in range(n)
            )
        )
        model.addConstrs(xi[i] >= 0 for i in range(n))

        # ------- objective --------
        model.setObjective(sum(xi[i] for i in range(n)), GRB.MINIMIZE)

        # optimize
        model.optimize()

        # get optimized values
        opt_beta = [beta[j].x for j in range(len(beta))]
        opt_b = b.x

        # convert to numpy
        beta = np.array(opt_beta).reshape(-1, 1)
        b = opt_b
        return beta, b
