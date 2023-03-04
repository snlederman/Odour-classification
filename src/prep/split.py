"""
splitting data for modeling
"""

# packages
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from cmd_parse import get_args

def main():
    """program skeleton"""
    args = get_args()
    
    # loading data
    if args["augment"]:
        labels_path = os.path.join(PROJECT_DIR,"data", "augmented", "labels.csv")
        features_path = os.path.join(PROJECT_DIR,"data", "augmented", "features.csv")
        features_scaled_path = os.path.join(PROJECT_DIR,"data", "augmented", "scaled", "features.csv")
    else:
        labels_path = os.path.join(PROJECT_DIR,"data", "cleaned", "labels.csv")
        features_path = os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv")
        features_scaled_path = os.path.join(PROJECT_DIR,"data", "cleaned", "scaled", "features.csv")


    labels = pd.read_csv(labels_path)
    features = pd.read_csv(features_path)
    features_scaled = pd.read_csv(features_scaled_path)

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

    # saving data
    if args["augment"]:
        labels_train_path = os.path.join(PROJECT_DIR,"data", "splitted", "train", "augmented", "labels.csv")
        features_train_path = os.path.join(PROJECT_DIR,"data", "splitted", "train", "augmented", "features.csv")
        features_train_scaled_path = os.path.join(PROJECT_DIR,"data", "splitted", "train", "augmented", "scaled", "features.csv")
        
        labels_test_path = os.path.join(PROJECT_DIR,"data", "splitted", "test", "augmented", "labels.csv")
        features_test_path = os.path.join(PROJECT_DIR,"data", "splitted", "test", "augmented", "features.csv")
        features_test_scaled_path = os.path.join(PROJECT_DIR,"data", "splitted", "test", "augmented", "scaled", "features.csv")
            
    else:
        labels_train_path = os.path.join(PROJECT_DIR,"data", "splitted", "train", "labels.csv")
        features_train_path = os.path.join(PROJECT_DIR,"data", "splitted", "train", "features.csv")
        features_train_scaled_path = os.path.join(PROJECT_DIR,"data", "splitted", "train", "scaled", "features.csv")
        
        labels_test_path = os.path.join(PROJECT_DIR,"data", "splitted", "test", "labels.csv")
        features_test_path = os.path.join(PROJECT_DIR,"data", "splitted", "test", "features.csv")
        features_test_scaled_path = os.path.join(PROJECT_DIR,"data", "splitted", "test", "scaled", "features.csv")
            

    y_train.to_csv(labels_train_path, index=False)
    X_train.to_csv(features_train_path, index=False)
    X_train_scaled.to_csv(features_train_scaled_path, index=False)

    y_test.to_csv(labels_test_path, index=False)
    X_test.to_csv(features_test_path, index=False)
    X_test_scaled.to_csv(features_test_scaled_path, index=False)


if __name__ == "__main__":
    main()
