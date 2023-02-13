"""
random forest model
"""

# packages
import os
import sys
# import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from summeries_classification import summeries_multiclass_report


def main():
    """program skeleton"""

    # loading training data
    y_train = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "train", "labels.csv")).set_index("ID")
    X_train = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "train", "features.csv")).set_index("ID")

    # loading testing data
    y_test = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "test", "labels.csv")).set_index("ID")
    X_test = pd.read_csv(os.path.join(PROJECT_DIR,"data", "splitted", "test", "features.csv")).set_index("ID")

    model = RandomForestClassifier()
    model.fit(X_train, y_train["label"])
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_summary = summeries_multiclass_report(report)
    
    return report_summary


if __name__ == "__main__":
    print(main())
