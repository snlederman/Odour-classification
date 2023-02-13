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
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from summeries_classification import summeries_multiclass_report


class BaselineModel:
    """random sampler from train labels"""
    def __init__(self):
        self.labels = None
        
    def fit(self, X, y):
        """recives pandas series as training labels and store them for predictions"""
        self.labels = y.values

    def predict(self, X):
        """predict random labels from training set at the length of input"""
        res = np.random.choice(self.labels, X.shape[0])
        return res


def main():
    """program skeleton"""

    # loading training data
    y_train = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "train", "labels.csv")).set_index("ID")
    X_train = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "train", "features.csv")).set_index("ID")

    # loading testing data
    y_test = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "test", "labels.csv")).set_index("ID")
    X_test = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "test", "features.csv")).set_index("ID")

    baseline_model = BaselineModel()
    baseline_model.fit(X_train, y_train["label"])
    y_pred = baseline_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_summary = summeries_multiclass_report(report)
    return report_summary


if __name__ == "__main__":
    print(main())