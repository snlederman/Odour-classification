"""
loads the data set
"""
# package
import os
import pandas as pd

def load_data(project_dir):
    """loads cleaned data"""
    labels_path = os.path.join(project_dir, "data", "cleaned", "labels.csv")
    features_path = os.path.join(project_dir, "data", "cleaned", "features.csv")

    labels =  pd.read_csv(labels_path).set_index("ID")
    features =  pd.read_csv(features_path).set_index("ID")

    return labels, features
