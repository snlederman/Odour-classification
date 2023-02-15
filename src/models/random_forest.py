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
from cmd_parse import get_args
from load_data import load_data
from summeries_classification import summeries_multiclass_report
from log_classification import log_metrics

def main():
    """program skeleton"""
    args = get_args()
    
    X_train, y_train, X_test, y_test = load_data(PROJECT_DIR, args["scale"])
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train["label"])
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    log_metrics(report, model, PROJECT_DIR, args["scale"])
    report_summary = summeries_multiclass_report(report)
    
    return report_summary


if __name__ == "__main__":
    print(main())
