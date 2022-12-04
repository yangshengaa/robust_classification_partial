"""parsing real dataset from UCI"""

# load packages
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ======== arguments ========
parser = argparse.ArgumentParser()

parser.add_argument(
    "--data",
    nargs="+",
    default=["australian", "bands", "heart", "hepatitis", "horse"],
    type=str,
    help="select the type of real dataset to parse",
)
parser.add_argument(
    "--seed", default=42, type=int, help="the seed for probabilistic imputation"
)

args = parser.parse_args()

SEED = args.seed
np.random.seed(SEED)


def parse_australian():
    """australian credit dataset"""
    # load data
    path = "data/raw/australian/australian.dat"
    df = pd.read_csv(path, sep=" ", header=None)

    # get columns
    p = df.shape[1]
    label_col = [14]
    cat_cols = [0, 3, 4, 5, 7, 6, 10, 11]  # categorical columns
    num_cols = list(set(range(p)) - set(label_col + cat_cols))

    # get data
    X_num = df[num_cols].to_numpy().astype(float)
    X_num = (X_num - X_num.mean(axis=0, keepdims=True)) / X_num.std(
        axis=0, keepdims=True
    )
    X_cat = OneHotEncoder(drop="first", sparse=False).fit_transform(df[cat_cols])
    y = ((df[label_col].to_numpy() == 1) * 2 - 1).astype(int).flatten()
    X = np.hstack((X_num, X_cat))

    # save data
    folder_path = "data/real_australian"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    np.save(os.path.join(folder_path, "X.npy"), X)
    np.save(os.path.join(folder_path, "y.npy"), y)
    np.save(os.path.join(folder_path, "selected_terms.npy"), np.array(range(6)))


def parse_bands():
    """band dataset"""
    path = "data/raw/bands/bands.data"
    df = pd.read_csv(path, header=None)
    # dropna
    df = df.loc[(df == "?").mean(axis=1) < 0.05]
    y = ((df.iloc[:, -1] == "band") * 2 - 1).to_numpy().astype(int).flatten()

    df = df.iloc[:, [4, 5, 8, 9, 10, 11, 21, 24, 27, 29, 30, 32, 34, 35, 36, 37, 38]]
    X_num = df.iloc[:, 6:].to_numpy().astype(float)
    X_num = (X_num - X_num.mean(axis=0, keepdims=True)) / X_num.std(
        axis=0, keepdims=True
    )
    X_cat = OneHotEncoder(drop="first", sparse=False).fit_transform(df.iloc[:, :6])
    X = np.hstack((X_num, X_cat))

    # save data
    folder_path = "data/real_bands"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    np.save(os.path.join(folder_path, "X.npy"), X)
    np.save(os.path.join(folder_path, "y.npy"), y)
    np.save(os.path.join(folder_path, "selected_terms.npy"), np.array(range(11)))


def parse_heart():
    """heart dataset"""
    path = "data/raw/heart/heart.dat"
    df = pd.read_csv(path, sep=" ", header=None)
    y = ((df.iloc[:, -1].to_numpy() == 2) * 2 - 1).astype(int).flatten()
    cat_cols = [1, 5, 8, 6, 2, 12, 10]
    num_cols = [0, 3, 4, 7, 9, 11]

    # get X and y
    X_num = df.iloc[:, num_cols].to_numpy().astype(float)
    X_num = (X_num - X_num.mean(axis=0, keepdims=True)) / X_num.std(
        axis=0, keepdims=True
    )
    X_cat = OneHotEncoder(drop="first", sparse=False).fit_transform(
        df.iloc[:, cat_cols]
    )
    X = np.hstack((X_num, X_cat))

    # save data
    folder_path = "data/real_heart"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    np.save(os.path.join(folder_path, "X.npy"), X)
    np.save(os.path.join(folder_path, "y.npy"), y)
    np.save(os.path.join(folder_path, "selected_terms.npy"), np.array(range(6)))


def parse_hepatitis():
    """hepatitis dataset"""
    path = "data/raw/hepatitis/hepatitis.data"
    df = pd.read_csv(path, header=None)
    y = ((df.iloc[:, -1] == 1) * 2 - 1).to_numpy().astype(int)
    X_cat = OneHotEncoder(drop="first", sparse=False).fit_transform(
        df.iloc[:, [2, 4, 19]]
    )
    X_num = df.iloc[:, [1]].to_numpy().astype(float)
    X_num = (X_num - X_num.mean(axis=0, keepdims=True)) / X_num.std(
        axis=0, keepdims=True
    )

    X = np.hstack((X_num, X_cat))

    # save data
    folder_path = "data/real_hepatitis"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    np.save(os.path.join(folder_path, "X.npy"), X)
    np.save(os.path.join(folder_path, "y.npy"), y)
    np.save(os.path.join(folder_path, "selected_terms.npy"), np.array(range(1)))


def parse_horse():
    """horse dataaset"""
    path = "data/raw/horse/horse-colic.data"
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.strip().split())
    df = pd.DataFrame(data)

    # probabalistic imputation
    df_copy = df.copy()
    for col in df_copy.columns:
        col_candidates = [x for x in df_copy[col].tolist() if x != "?"]
        for row in range(df.shape[0]):
            if "?" == df.loc[row, col]:
                df_copy.loc[row, col] = np.random.choice(col_candidates)

    # get y
    y = ((df_copy[23] == "1") * 2 - 1).to_numpy().astype(int).flatten()

    # get x
    # full list
    # cat_cols = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 20, 24, 25, 26, 27]
    # selected partial list
    cat_cols = [8, 9, 10, 11, 12, 13, 14, 16, 17, 20, 25, 26, 27]
    num_cols = [3, 4, 5, 15, 18, 19, 21]

    X_num = df_copy[num_cols].to_numpy().astype(float)
    X_num = (X_num - X_num.mean(axis=0, keepdims=True)) / X_num.std(
        axis=0, keepdims=True
    )
    X_cat = OneHotEncoder(drop="first", sparse=False).fit_transform(df_copy[cat_cols])
    X = np.hstack((X_num, X_cat))

    # save data
    folder_path = "data/real_horse"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    np.save(os.path.join(folder_path, "X.npy"), X)
    np.save(os.path.join(folder_path, "y.npy"), y)
    np.save(os.path.join(folder_path, "selected_terms.npy"), np.array(range(7)))


def main():
    """main driver"""
    if "australian" in args.data:
        print("parsing australian")
        parse_australian()
    if "bands" in args.data:
        print("parsing bands")
        parse_bands()
    if "heart" in args.data:
        print("parsing heart")
        parse_heart()
    if "hepatitis" in args.data:
        print("parsing hepatitis")
        parse_hepatitis()
    if "horse" in args.data:
        print("parsing horse")
        parse_horse()


if __name__ == "__main__":
    main()
