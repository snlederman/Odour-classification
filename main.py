"""
main script to run pipline and models
"""

# packages
import os
import sys

SCRIPT_PATH = os.path.realpath(__file__)

# seats in the main entry to the repo, removing only file name
SCRIPT_NAME = "main.py"
PROJECT_DIR = SCRIPT_PATH[:-len(SCRIPT_NAME)]

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from cmd_parse import get_args
from load_data import load_data

sys.path.append(os.path.join(PROJECT_DIR, "src", "prep"))
from scale import scale
from clip import clip
from derive import derive
from reduce import reduce
from split import split

sys.path.append(os.path.join(PROJECT_DIR, "src", "models"))
from most_common import most_common
from random_sampler import random_sampler
from logistic_regression import logistic_regression
from random_forest import random_forest
from gradient_boosting import gradient_boosting
from ada_boost import ada_boost
from dense_neuralnet import dense_neuralnet
from MLP import MLP
from RNN import RNN
from KNN import KNN


def main():
    """program skeleton"""

    args = get_args()

    # load cleaned data
    labels, features = load_data(PROJECT_DIR)

    # pre-split preps
    pre_preps = ["scale", "clip", "derive"]
    for key, value in args.items():
        if key in pre_preps and value:
            features = eval(f"{key}({features})")

    # split train test
    x_train, y_train, x_test, y_test = split(labels, features)

    # post-split preps
    post_preps = ["reduce"]
    for key, value in args.items():
        if key in post_preps and value:
            x_train = eval(f"{key}({y_train}, {x_train})")
            x_test = eval(f"{key}({y_test}, {x_test})")

    # modeling
    results = eval(f"{args['model']}({x_train}, {y_train}, {x_test}, {y_test})")

    # saving metrics

if __name__ == "__main__":
    main()
