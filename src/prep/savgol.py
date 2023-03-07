"""
feature extraction using savgol filter
"""

# packages
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from cmd_parse import get_args, args_to_path
from load_data import load_data


def main():
    """program skeleton"""
    args = get_args()

    data_path = args_to_path(args)

    # loading data
    X_train, y_train, X_test, y_test = load_data(PROJECT_DIR, args)

    X_train_first = pd.DataFrame(savgol_filter(X_train, deriv=1, window_length=3, polyorder=1, axis=0), columns=X_train.columns, index=X_train.index)
    X_train_second = pd.DataFrame(savgol_filter(X_train, deriv=2, window_length=3, polyorder=2, axis=0), columns=X_train.columns, index=X_train.index)
    
    X_test_first = pd.DataFrame(savgol_filter(X_test, deriv=1, window_length=3, polyorder=1, axis=0), columns=X_test.columns, index=X_test.index)
    X_test_second = pd.DataFrame(savgol_filter(X_test, deriv=2, window_length=3, polyorder=2, axis=0), columns=X_test.columns, index=X_test.index)

    X_train = pd.concat([X_train, X_train_first, X_train_second], axis=1)
    X_test = pd.concat([X_test, X_test_first, X_test_second], axis=1)

    model = RandomForestClassifier()
    
    model.fit(X_train, y_train["label"])
    y_pred = model.predict(X_test)
        
    model.fit(X_train_first, y_train["label"])
    y_pred = model.predict(X_test_first)

    model.fit(X_train_second, y_train["label"])
    y_pred = model.predict(X_test_second)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    log_metrics(report, model, PROJECT_DIR, args["scaled"], args["augmented"])
    report_summary = summeries_multiclass_report(report)
    print(report_summary)
    confusion_matrix(y_test, y_pred)
    

if __name__ == "__main__":
    main()
