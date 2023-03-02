"""
loads the data needed for model training and testing
"""
# package
import os
import pandas as pd

def load_data(PROJECT_DIR, scaled, augmented):
    
    if augmented:
        if scaled:
            labels_train_path = os.path.join(PROJECT_DIR, "data", "splitted", "train", "augmented", "labels.csv")
            labels_test_path = os.path.join(PROJECT_DIR, "data", "splitted", "test", "augmented", "labels.csv")
            
            features_train_path = os.path.join(PROJECT_DIR, "data", "splitted", "train", "augmented", "scaled", "features.csv")
            features_test_path = os.path.join(PROJECT_DIR, "data", "splitted", "test", "augmented", "scaled", "features.csv")
        else:
            labels_train_path = os.path.join(PROJECT_DIR, "data", "splitted", "train", "augmented", "labels.csv")
            labels_test_path = os.path.join(PROJECT_DIR, "data", "splitted", "test", "augmented", "labels.csv")
            
            features_train_path = os.path.join(PROJECT_DIR, "data", "splitted", "train", "augmented", "features.csv")
            features_test_path = os.path.join(PROJECT_DIR, "data", "splitted", "test", "augmented", "features.csv")
    else:
        if scaled:
            labels_train_path = os.path.join(PROJECT_DIR, "data", "splitted", "train", "labels.csv")
            labels_test_path = os.path.join(PROJECT_DIR, "data", "splitted", "test", "labels.csv")
            
            features_train_path = os.path.join(PROJECT_DIR, "data", "splitted", "train", "scaled", "features.csv")
            features_test_path = os.path.join(PROJECT_DIR, "data", "splitted", "test", "scaled", "features.csv")
        else:
            labels_train_path = os.path.join(PROJECT_DIR, "data", "splitted", "train", "labels.csv")
            labels_test_path = os.path.join(PROJECT_DIR, "data", "splitted", "test", "labels.csv")
            
            features_train_path = os.path.join(PROJECT_DIR, "data", "splitted", "train", "features.csv")
            features_test_path = os.path.join(PROJECT_DIR, "data", "splitted", "test", "features.csv")
    
    # loading labels
    y_train = pd.read_csv(labels_train_path).set_index("ID")
    y_test = pd.read_csv(labels_test_path).set_index("ID")
        
    # loading features
    X_train = pd.read_csv(features_train_path).set_index("ID")
    X_test = pd.read_csv(features_test_path).set_index("ID")
        
    return  X_train, y_train, X_test, y_test
