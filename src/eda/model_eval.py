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
    labels = ["1-Benz", "2-Hex", "3-Ethyl", "4-Rose", "5-Lem", "6-Ger", "7-Cit", "8-Van"]
    metrics.drop(columns=labels, inplace=True)
    to_drop = ["log_id", "fourier", "metric", "macro avg", "weighted avg"]
    metrics.drop(columns=to_drop, inplace=True)

    metrics.sort_values("accuracy", ascending=False, inplace=True)
    metrics.loc[metrics["metric"] == "f1-score",labels].iloc[0]


if __name__ == "__main__":
    print(main())
