"""
loads the data needed for model training and testing
"""
# package
import os
import pandas as pd

def load_data(PROJECT_DIR, scaled):

    # loading labels
    y_train = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "train", "labels.csv")).set_index("ID")
    y_test = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "test", "labels.csv")).set_index("ID")

    # loading features
    if scaled:
        X_train = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "train", "scaled", "features.csv")).set_index("ID")
        X_test = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "test", "scaled", "features.csv")).set_index("ID")
    else:
        X_train = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "train", "features.csv")).set_index("ID")
        X_test = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "test", "features.csv")).set_index("ID")

    return  X_train, y_train, X_test, y_test
