"""
loads the data needed for model training and testing
"""
# package
import os
import pandas as pd

from cmd_parse import args_to_path

def load_data(project_dir, args):
    
    data_path = args_to_path(args)
    
    labels_train_path = os.path.join(project_dir, "data", "cleaned", "splitted", "train", data_path, "labels.csv")
    labels_test_path = os.path.join(project_dir, "data", "cleaned", "splitted", "test", data_path, "labels.csv")
    
    features_train_path = os.path.join(project_dir, "data", "cleaned", "splitted", "train", data_path, "features.csv")
    features_test_path = os.path.join(project_dir, "data", "cleaned", "splitted", "test", data_path, "features.csv")
    
    # loading labels
    y_train = pd.read_csv(labels_train_path).set_index("ID")
    y_test = pd.read_csv(labels_test_path).set_index("ID")
        
    # loading features
    X_train = pd.read_csv(features_train_path).set_index("ID")
    X_test = pd.read_csv(features_test_path).set_index("ID")
        
    return  X_train, y_train, X_test, y_test
