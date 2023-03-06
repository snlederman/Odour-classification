"""
base line model to compare any new model to
"""

# packages
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from cmd_parse import get_args
from load_data import load_data
from summeries_classification import summeries_multiclass_report
from log_classification import log_metrics


class MostCommon:
    """predicting alwase the most common label in training data"""
    def __init__(self):
        self.labels = None
        
    def fit(self, X, y):
        """recives pandas series as training labels and store them for predictions"""
        most_common_label = y.value_counts().index[0]
        self.labels = most_common_label

    def predict(self, X):
        """predict random labels from training set at the length of input"""
        res = np.array([self.labels] * X.shape[0]).astype(object)
        return res


def main():
    """program skeleton"""
    args = get_args()
    
    X_train, y_train, X_test, y_test = load_data(PROJECT_DIR, args["scale"], args["augment"])

    model = MostCommon()
    model.fit(X_train, y_train["label"])
    y_pred = model.predict(X_test)

    report = classification_report(y_test["label"], y_pred, output_dict=True)
    log_metrics(report, model, PROJECT_DIR, args["scale"], args["augment"])
    report_summary = summeries_multiclass_report(report)
    
    return report_summary


if __name__ == "__main__":
    print(main())
