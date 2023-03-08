"""
splitting data for modeling
"""

# packages
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from cmd_parse import get_args, args_to_path

def main():
    """program skeleton"""
    args = get_args()
    
    data_path = args_to_path(args)
    
    # loading data
    labels_path = os.path.join(PROJECT_DIR, "data", "cleaned", data_path, "labels.csv")
    features_path = os.path.join(PROJECT_DIR, "data", "cleaned", data_path, "features.csv")
    
    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)

    unique_IDs = labels["ID"].unique()
    # split to 80% training (later 60-20 with validation) and 20% testing
    train_num = int(len(unique_IDs) * 0.8)
    train_IDs = np.random.choice(unique_IDs, train_num, replace=False)
    test_IDs = unique_IDs[~np.isin(unique_IDs, train_IDs)]

    y_train = labels.loc[np.isin(labels["ID"],train_IDs),:]
    X_train = features.loc[np.isin(features["ID"],train_IDs),:]

    y_test = labels.loc[np.isin(labels["ID"],test_IDs),:]
    X_test = features.loc[np.isin(features["ID"],test_IDs),:]

    # saving data
    labels_train_path = os.path.join(PROJECT_DIR, "data", "cleaned", "splitted", "train", data_path, "labels.csv")
    features_train_path = os.path.join(PROJECT_DIR, "data", "cleaned", "splitted", "train", data_path, "features.csv")
    
    labels_test_path = os.path.join(PROJECT_DIR, "data", "cleaned", "splitted", "test", data_path, "labels.csv")
    features_test_path = os.path.join(PROJECT_DIR, "data", "cleaned", "splitted", "test", data_path, "features.csv")

    y_train.to_csv(labels_train_path, index=False)
    X_train.to_csv(features_train_path, index=False)
    
    y_test.to_csv(labels_test_path, index=False)
    X_test.to_csv(features_test_path, index=False)


if __name__ == "__main__":
    main()
