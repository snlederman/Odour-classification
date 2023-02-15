"""
splitting data for modeling
"""

# packages
import os
import numpy as np
import pandas as pd

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]


def main():
    """program skeleton"""

    labels = pd.read_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "labels.csv"))
    features = pd.read_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv"))
    features_scaled = pd.read_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "scaled", "features.csv"))

    unique_IDs = labels["ID"].unique()
    # split to 80% training (later 60-20 with validation) and 20% testing
    train_num = int(len(unique_IDs) * 0.8)
    train_IDs = np.random.choice(unique_IDs, train_num, replace=False)
    test_IDs = unique_IDs[~np.isin(unique_IDs, train_IDs)]

    y_train = labels.loc[np.isin(labels["ID"],train_IDs),:]
    X_train = features.loc[np.isin(features["ID"],train_IDs),:]
    X_train_scaled = features_scaled.loc[np.isin(features_scaled["ID"],train_IDs),:]

    y_test = labels.loc[np.isin(labels["ID"],test_IDs),:]
    X_test = features.loc[np.isin(features["ID"],test_IDs),:]
    X_test_scaled = features_scaled.loc[np.isin(features_scaled["ID"],test_IDs),:]

    y_train.to_csv(os.path.join(PROJECT_DIR,"data", "splitted", "train", "labels.csv"), index=False)
    X_train.to_csv(os.path.join(PROJECT_DIR,"data", "splitted", "train", "features.csv"), index=False)
    X_train_scaled.to_csv(os.path.join(PROJECT_DIR,"data", "splitted", "train", "scaled", "features.csv"), index=False)

    y_test.to_csv(os.path.join(PROJECT_DIR,"data", "splitted", "test", "labels.csv"), index=False)
    X_test.to_csv(os.path.join(PROJECT_DIR,"data", "splitted", "test", "features.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(PROJECT_DIR,"data", "splitted", "test", "scaled", "features.csv"), index=False)


if __name__ == "__main__":
    main()
