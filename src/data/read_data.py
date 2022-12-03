"""
methods to load synthetic and real dataset
"""

# load packages
import os
from typing import Tuple
import numpy as np

# path
DATA_PATH = "data/"


def load_syn(folder_name: str) -> Tuple[np.ndarray]:
    """load synthetic data by folder name"""
    folder_path = os.path.join(DATA_PATH, folder_name)

    # load data
    X_train = np.load(os.path.join(folder_path, "X_train.npy"))
    X_val = np.load(os.path.join(folder_path, "X_val.npy"))
    X_test = np.load(os.path.join(folder_path, "X_test.npy"))
    y_train = np.load(os.path.join(folder_path, "y_train.npy"))
    y_val = np.load(os.path.join(folder_path, "y_val.npy"))
    y_test = np.load(os.path.join(folder_path, "y_test.npy"))

    # load underlying decision boundary
    beta_true = np.load(os.path.join(folder_path, "beta_true.npy"))
    b_true = np.load(os.path.join(folder_path, "b_true.npy"))[0]

    return X_train, X_val, X_test, y_train, y_val, y_test, beta_true, b_true


def load_real(folder_name: str) -> Tuple[np.ndarray]:
    """load real dataset"""
    raise NotImplementedError()
