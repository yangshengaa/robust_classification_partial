"""train synthetic dataset and log model parameters"""

# load packages
import os
import numpy as np
import argparse

# load file
from models import LRRegular, LRRobust
from data import load_syn
from evaluation import *

# =========== arguments ==============
parser = argparse.ArgumentParser()

# data
parser.add_argument(
    "--data",
    default="syn_cont",
    type=str,
    help="the folder name of the synthetic dataset",
)

# model
parser.add_argument(
    "--model", default="LRRegular", type=str, help="the type of model to train"
)
parser.add_argument(
    "--lr", default=1e-3, type=float, help="LR specific: learning rate of SGD"
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
    default=[0, 1e-3, 1e-2, 1e-1, 1, 10],
    nargs="+",
    help="the list of penalization terms to try",
)
parser.add_argument(
    "--terms",
    default=None,
    nargs="+",
    help="the index selected to robustify/penalize. Default penalizes all",
)

args = parser.parse_args()

# ========= load data ==========
X_train, X_val, X_test, y_train, y_val, y_test, beta_true, b_true = load_syn(args.data)
selected_terms = list(range(X_train.shape[1])) if args.terms is None else args.terms

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
        raise NotImplementedError()
    elif args.model == "SVMRobust":
        raise NotImplementedError()
    else:
        raise NotImplementedError()


# ========= training =============
def train():
    # tune hyperparameters
    best_beta, best_b = None, None
    best_model = None
    best_gamma = None
    best_acc = -float("inf")
    for gamma in args.gamma_list:
        model = select_model(gamma)
        model.fit(X_train, y_train)
        train_predict = model.predict(X_train)
        train_acc = (train_predict == y_train).mean()
        val_predict = model.predict(X_val)
        val_acc = (val_predict == y_val).mean()

        print(f"gamma {gamma:.4f}: train acc {train_acc:.4f}, validation acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_beta = (model.beta,)
            best_b = model.b
            best_gamma = gamma
            best_model = model
            best_acc = val_acc

    print(best_beta, best_b)
    # test
    test_predict = best_model.predict(X_test)
    test_acc = (test_predict == y_test).mean()
    test_norm_deviance = norm_deviance(best_beta, best_b, beta_true, b_true)
    print(
        f"selected gamma {best_gamma:.4f}, test acc: {test_acc:.4f}, test norm deviance: {test_norm_deviance:.4f}"
    )

    return best_gamma, test_acc, test_norm_deviance


def log_performance(best_gamma: float, test_acc: float, test_norm_deviance: float):
    """append results to file"""
    msg = "{},{},{},{},{},{}".format(
        args.data,
        args.model,
        '"' + str(selected_terms) + '"',
        best_gamma,
        test_acc,
        test_norm_deviance,
    )
    with open("results/syn_performance.txt", "a+") as f:
        f.write(msg)
        f.write("\n")


def main():
    """run training and logging"""
    best_gamma, test_acc, test_norm_deviance = train()
    log_performance(best_gamma, test_acc, test_norm_deviance)


if __name__ == "__main__":
    main()
