"""
robust logistic regression with l2 penalties
implemented using autodiff library (e.g. PyTorch) since this is an unconstrained optimization problem
"""

# load packages
import os
from typing import List
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import SGD

# initialization parameters
INIT_MEAN = 0
INIT_STD = 1

# ---------------------------------------------
# =============== base class ==================
# ---------------------------------------------
class LRBase(nn.Module):
    def __init__(self, p: int, gamma: float, selected_feature_list: List):
        """
        :param p: the number of parameters
        :param selected_feature_list: the index to penalize
        """
        super().__init__()
        self.gamma = gamma
        self.selected_feature_list = selected_feature_list
        # init model
        self.beta = Parameter(torch.normal(INIT_MEAN, INIT_STD, size=(p, 1)).double())
        self.b = Parameter(torch.normal(INIT_MEAN, INIT_STD, size=(1,)).double())

    def _get_penalty(self, beta: torch.Tensor):
        """compute l2 penalty term with selected parameters"""
        penalty = (beta[self.selected_feature_list] ** 2).sum().sqrt() * self.gamma
        return penalty

    def _get_loss(
        self, X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, b: torch.Tensor
    ):
        """compute loss in different manners"""
        raise NotImplementedError()

    def forward(self, X: torch.Tensor, y: torch.Tensor):
        """return loss for optimization"""
        loss = self._get_loss(X, y, self.beta, self.b)
        return loss


class LRRegularizedBase(LRBase):
    def _get_loss(
        self, X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, b: torch.Tensor
    ):
        """compute loss of normally regularized logistic regression"""
        penalty = self._get_penalty(beta)
        loss_with_penalty = torch.sum(
            torch.log(torch.exp(-(X @ beta + b) * y) + 1) + penalty
        )
        return loss_with_penalty


class LRRobustifiedBase(LRBase):
    def _get_loss(
        self, X: torch.Tensor, y: torch.Tensor, beta: torch.Tensor, b: torch.Tensor
    ):
        """compute loss of normally robustified logistic regression"""
        penalty = self._get_penalty(beta)
        loss_with_penalty = torch.sum(
            torch.log(torch.exp(-(X @ beta + b) * y + penalty) + 1)
        )
        return loss_with_penalty


# ------------------------------------------------
# ================ user interface ================
# ------------------------------------------------
class LRUser:
    """an sklearn-like interface, with fit and predict"""

    def __init__(
        self,
        gamma=0.01,
        selected_feature_list=None,
        epochs=3000,
        lr=1e-2,
        verbose=False,
    ):
        """
        :param gamma: the regularization parameter
        :param selected_feature_list: the index list for selected perturbations
        :param epochs: the number of epochs to train
        :param lr: the learning rate of SGD
        :param verbose: True to print out training loss
        """
        self.gamma = gamma
        self.selected_feature_list = selected_feature_list

        # other hyperparameters
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

    def _select_model(
        self, p: int, gamma: float, selected_feature_list: List
    ) -> nn.Module:
        """select and initialize appropriate model"""
        raise NotImplementedError()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        training of logistic regression
        :param X: training predictors
        :param y: training responses
        """
        # convert to tensors
        n, p = X.shape
        X_tensor = torch.tensor(X).reshape(n, p)
        y_tensor = torch.tensor(y).reshape(n, 1)

        # initialize model
        model = self._select_model(p, self.gamma, self.selected_feature_list)

        # initialize optimizer
        opt = SGD(model.parameters(), lr=self.lr)

        # training
        model.train()
        for e in range(self.epochs + 1):
            opt.zero_grad()
            loss = model(X_tensor, y_tensor)
            loss.backward()
            opt.step()

            # print loss
            if self.verbose and e % 1 == 0:
                print("epoch {:3d}: loss {:.4f}".format(e, loss.item()))

        # obtain parameters and cast to numpy
        self.beta = model.beta.detach().cpu().numpy()
        self.b = model.b.item()

    def _decision_rule(self, X: np.ndarray, beta: np.ndarray, b: float):
        """give decision based on decision boundary"""
        raw = (X @ beta + b).flatten()
        decision = ((raw > 0) * 2 - 1).astype(int)  # convert to 1 and -1
        return decision

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predict class labels by
        :param X: the test dataset
        """
        return self._decision_rule(X, self.beta, self.b)


class LRRegular(LRUser):
    """regularized logistic regression"""

    def _select_model(
        self, p: int, gamma: float, selected_feature_list: List
    ) -> nn.Module:
        return LRRegularizedBase(p, gamma, selected_feature_list)


class LRRobust(LRUser):
    """robustified logistic regression"""

    def _select_model(
        self, p: int, gamma: float, selected_feature_list: List
    ) -> nn.Module:
        return LRRobustifiedBase(p, gamma, selected_feature_list)
