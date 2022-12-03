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
SEED = 20
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


def make_syn_discrete():
    """
    create a discrete version of synthetic dataset
    boundary: y = 5 x + 0.5, and y only takes values from {0, 1}
    """
    # train and validation data
    X0 = np.array([0] * 100 + [1] * 100)
    X1 = np.random.uniform(-0.5, 0.5, size=(200,))
    X_train = np.vstack((X1, X0)).T
    y_train = ((X_train[:, 0] * 5 + 0.5 < X_train[:, 1]) * 2 - 1).astype(int).flatten()

    # validation
    X0 = np.array([0] * 20 + [1] * 20)
    X1 = np.random.uniform(-0.5, 0.5, size=(40,))
    X_val = np.vstack((X1, X0)).T
    y_val = ((X_val[:, 0] < 0) * 2 - 1).astype(int).flatten()

    # test
    X0 = np.array([0] * 5000 + [1] * 5000)
    X1 = np.random.uniform(-0.5, 0.5, size=(10000,))
    X_test = np.vstack((X1, X0)).T
    y_test = ((X_test[:, 0] * 5 + 0.5 < X_test[:, 1]) * 2 - 1).astype(int).flatten()
    # corrupt 5% of the testing
    corrupt_index = np.random.choice(list(range(10000)), size=(500))
    y_test[corrupt_index] = -y_test[corrupt_index]

    # save data
    folder = os.path.join(DATA_PATH, "syn_dis")
    if not os.path.exists(folder):
        os.mkdir(folder)

    np.save(os.path.join(folder, "X_train.npy"), X_train)
    np.save(os.path.join(folder, "X_val.npy"), X_val)
    np.save(os.path.join(folder, "X_test.npy"), X_test)
    np.save(os.path.join(folder, "y_train.npy"), y_train)
    np.save(os.path.join(folder, "y_val.npy"), y_val)
    np.save(os.path.join(folder, "y_test.npy"), y_test)

    # log underlying decision boundary
    beta_true = np.array([[-5], [1]])
    b_true = np.array([-0.5])
    np.save(os.path.join(folder, "beta_true.npy"), beta_true)
    np.save(os.path.join(folder, "b_true.npy"), b_true)

    # make plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="tab:red")
    ax.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], c="tab:blue")
    ax.scatter(X_val[y_val == 1, 0], X_val[y_val == 1, 1], c="tab:red", marker="+")
    ax.scatter(X_val[y_val == -1, 0], X_val[y_val == -1, 1], c="tab:blue", marker="+")

    ax.plot([-0.2, 0.2], [-0.5, 1.5], c="black", linestyle="--")  # visualize boundary

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
    make_syn_discrete()
