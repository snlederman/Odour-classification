"""
Multi-layer Perceptron classifier.
"""

# packages
import os
import sys
# import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def main():
    """program skeleton"""
    args = get_args()
    
    X_train, y_train, X_test, y_test = load_data(PROJECT_DIR, args)
    
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs')
    model.fit(X_train, y_train["label"])
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    # log_metrics(report, model, PROJECT_DIR, args["scaled"], args["augmented"])
    report_summary = summeries_multiclass_report(report)
    
    return report_summary


if __name__ == "__main__":
    print(main())