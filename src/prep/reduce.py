"""
feature extraction using savgol filter
"""

# packages
import os
import sys
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from cmd_parse import get_args, args_to_path
from load_data import load_data

def main():
    """program skeleton"""
    args = get_args()
    
    data_path = args_to_path(args)

    # loading data
    X_train, y_train, X_test, y_test = load_data(PROJECT_DIR, args)
    
    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    lda.fit(X_train, np.ravel(y_train))
    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)

    # saving data
    features_train_path = os.path.join(PROJECT_DIR, "data", "cleaned", "splitted", "train", data_path, "reduced", "features.csv")
    features_test_path = os.path.join(PROJECT_DIR, "data", "cleaned", "splitted", "test", data_path, "reduced", "features.csv")

    X_train.to_csv(features_train_path, index=False)    
    X_test.to_csv(features_test_path, index=False)


if __name__ == "__main__":
    main()
