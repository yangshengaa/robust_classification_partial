"""
create synthetic dataset in Robust Classification
"""

# load packages
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.style.use("ggplot")

# fix seed
SEED = 404
np.random.seed(SEED)

# path
DATA_PATH = "data/"


def make_syn_cont():
    """create continuous synthetic dataset"""
    # train and validation
    pos_raw = np.random.normal([1.5, 1.5], 1, size=(25, 2))
    neg_raw = np.random.normal([-1.5, -1.5], 1, size=(25, 2))
    outliers_raw = np.random.normal(0, np.sqrt(3), size=(10, 2))
    X_raw = np.vstack((pos_raw, neg_raw, outliers_raw))
    y_raw = np.hstack(
        (np.array([1] * 25 + [-1] * 25), np.random.choice([-1, 1], 10, replace=True))
    )

    # 75 / 25 split
    X_train, X_val, y_train, y_val = train_test_split(
        X_raw, y_raw, test_size=0.25, stratify=y_raw, random_state=SEED
    )

    # create test dataset
    pos_test = np.random.normal([1.5, 1.5], 1, size=(10000, 2))
    neg_test = np.random.normal([-1.5, -1.5], 1, size=(10000, 2))
    X_test = np.vstack((pos_test, neg_test))
    y_test = np.array([1] * 10000 + [-1] * 10000)

    # save data
    folder = os.path.join(DATA_PATH, "syn_cont")
    if not os.path.exists(folder):
        os.mkdir(folder)

    np.save(os.path.join(folder, "X_train.npy"), X_train)
    np.save(os.path.join(folder, "X_val.npy"), X_val)
    np.save(os.path.join(folder, "X_test.npy"), X_test)
    np.save(os.path.join(folder, "y_train.npy"), y_train)
    np.save(os.path.join(folder, "y_val.npy"), y_val)
    np.save(os.path.join(folder, "y_test.npy"), y_test)

    # log underlying decision boundary
    beta_true = np.array([[1], [1]])
    b_true = np.array([0])
    np.save(os.path.join(folder, "beta_true.npy"), beta_true)
    np.save(os.path.join(folder, "b_true.npy"), b_true)

    # plot training and validation
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="tab:red")
    ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], c="tab:blue")
    ax.scatter(X_val[y_val == 1, 0], X_val[y_val == 1, 1], c="tab:red", marker="+")
    ax.scatter(X_val[y_val == -1, 0], X_val[y_val == -1, 1], c="tab:blue", marker="+")

    ax.plot([-4, 4], [4, -4], c="black", linestyle="--")  # visualize boundary

    ax.legend(["train pos", "train neg", "val pos", "val neg"])
    ax.set_title("training and validation")
    plt.savefig(
        os.path.join(folder, "train_val.png"),
        dpi=100,
        bbox_inches="tight",
        facecolor="white",
    )


if __name__ == "__main__":
    make_syn_cont()
