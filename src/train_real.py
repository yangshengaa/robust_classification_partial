"""real dataset training"""

# load packages
import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold


# load file
from models import LRRegular, LRRobust, SVMRegular, SVMRobust
from data import load_real
from evaluation import *

# =========== arguments ==============
parser = argparse.ArgumentParser()

# data
parser.add_argument(
    "--data",
    default="horse",
    type=str,
    help="the folder name of the real dataset",
)

# model
parser.add_argument(
    "--model",
    default="LRRegular",
    type=str,
    help="the type of model to train",
    choices=["LRRegular", "LRRobust", "SVMRegular", "SVMRobust"],
)
parser.add_argument("--folds", default=5, type=int, help="number of folds to evaluate")
parser.add_argument(
    "--lr", default=1e-2, type=float, help="LR specific: learning rate of SGD"
)
parser.add_argument(
    "--epochs", default=3000, type=int, help="LR specific: epochs to run SGD"
)
parser.add_argument(
    "--verbose", action="store_true", default=False, help="True to print training"
)

# penalization
parser.add_argument(
    "--gamma-list",
    default=[1e-3, 1e-2, 1e-1, 1, 10],
    nargs="+",
    type=float,
    help="the list of penalization terms to try",
)

# seed
parser.add_argument(
    "--seed", default=42, type=int, help="seed for train validation split"
)


args = parser.parse_args()

# ========= load data ==========
X, y, selected_terms = load_real("real_" + args.data)

# ========= init model ==========
def select_model(gamma: float):
    """return model with specified penalization"""
    if args.model == "LRRegular":
        return LRRegular(
            gamma, selected_terms, epochs=args.epochs, lr=args.lr, verbose=args.verbose
        )
    elif args.model == "LRRobust":
        return LRRobust(
            gamma, selected_terms, epochs=args.epochs, lr=args.lr, verbose=args.verbose
        )
    elif args.model == "SVMRegular":
        return SVMRegular(gamma, selected_terms, verbose=args.verbose)
    elif args.model == "SVMRobust":
        return SVMRobust(gamma, selected_terms, verbose=args.verbose)
    else:
        raise NotImplementedError(f"model {args.model} not defined")


# ========= training =============
def train():
    # tune hyperparameters
    best_gamma = None
    best_acc = -float("inf")
    best_acc_std = None

    for gamma in args.gamma_list:
        cur_val_accs = []
        # stratified K fold
        folds = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=args.seed
        ).split(X, y)
        print(f"======= gamma {gamma:.4f} =======")
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model = select_model(gamma)
            model.fit(X_train, y_train)
            train_predict = model.predict(X_train)
            train_acc = (train_predict == y_train).mean()
            val_predict = model.predict(X_val)
            val_acc = (val_predict == y_val).mean()

            print(
                f"fold {fold}: train acc {train_acc:.4f}, validation acc {val_acc:.4f}"
            )

            cur_val_accs.append(val_acc)

        # record
        if np.mean(cur_val_accs) > best_acc:
            best_gamma = gamma
            best_acc = np.mean(cur_val_accs)
            best_acc_std = np.std(cur_val_accs)

    # test
    print(
        f"selected gamma {best_gamma:.4f}, cv acc: {best_acc:.4f} +- {best_acc_std:.4f}"
    )

    return best_gamma, best_acc, best_acc_std


def log_performance(best_gamma: float, best_acc: float, best_acc_std: float):
    """append results to file"""
    msg = "{},{},{},{},{},{}".format(
        args.data,
        args.model,
        '"' + str(selected_terms) + '"',
        best_gamma,
        best_acc,
        best_acc_std,
    )
    with open("results/real_performance.txt", "a+") as f:
        f.write(msg)
        f.write("\n")


def main():
    """run training and logging"""
    best_gamma, best_acc, best_acc_std = train()
    log_performance(best_gamma, best_acc, best_acc_std)


if __name__ == "__main__":
    main()
