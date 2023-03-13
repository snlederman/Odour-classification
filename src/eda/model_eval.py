"""
evaluating models preformence
"""

# packages
import os
# import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

def main():
    """program skeleton"""
    metrics = pd.read_csv(os.path.join(PROJECT_DIR, "data", "stats", "metrics_log.csv"))
    metrics.dropna(inplace=True)
    metrics_short = metrics.copy()
    labels = ["1-Benz", "2-Hex", "3-Ethyl", "4-Rose", "5-Lem", "6-Ger", "7-Cit", "8-Van"]
    metrics_short.drop(columns=labels, inplace=True)
    to_drop = ["log_id", "fourier", "metric", "macro avg", "weighted avg"]
    metrics_short.drop(columns=to_drop, inplace=True)
    metrics_short.sort_values("accuracy", ascending=False, inplace=True)
    metrics_short = metrics_short.iloc[0::4,:]
    metrics_short.to_csv(os.path.join(PROJECT_DIR, "data", "stats", "accuracy_log.csv"), index=False)

    metrics.sort_values("accuracy", ascending=False, inplace=True)
    best_f1 = (
        metrics
        .loc[metrics["metric"] == "f1-score",labels]
        .iloc[0]
        .reset_index()
    )
    best_f1.rename(columns={"index":"label",best_f1.columns[1]:"F1-score"}, inplace=True)
    best_f1.to_csv(os.path.join(PROJECT_DIR, "data", "stats", "best_f1_log.csv"), index=False)
    


if __name__ == "__main__":
    main()
