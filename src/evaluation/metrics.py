"""evaluation metrics"""

# load packages
import numpy as np


def norm_deviance(
    beta_train: np.ndarray, b_train: float, beta_true: np.ndarray, b_true: float
):
    """get the difference in norm between the train boundary and true boundary"""
    # normalize
    beta_train_normalized = beta_train / np.linalg.norm(beta_train)
    beta_true_normalized = beta_true / np.linalg.norm(beta_true)

    # get deviance
    deviance = np.linalg.norm(beta_train_normalized - beta_true_normalized) + abs(
        b_train - b_true
    )
    return deviance
