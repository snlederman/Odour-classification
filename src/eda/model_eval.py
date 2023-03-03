"""
evaluating models preformence
"""

# packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

def main():
    """program skeleton"""
    metrics = pd.read_csv(os.path.join(PROJECT_DIR, "data", "metrics_log.csv"))
    metrics.columns
    rf_scaled_precision = metrics[
        (metrics["metric"] == "precision") &
        (metrics["model"] == "RandomForestClassifier") &
        (metrics["scaled"])
    ]
    
    p = rf_scaled_precision[["1-Benz","2-Hex","3-Ethyl","4-Rose","5-Lem","6-Ger","7-Cit","8-Van"]].mean(axis=0)
    
    rf_scaled_recall = metrics[
        (metrics["metric"] == "recall") &
        (metrics["model"] == "RandomForestClassifier") &
        (metrics["scaled"])
    ]
    
    s = rf_scaled_recall[["1-Benz","2-Hex","3-Ethyl","4-Rose","5-Lem","6-Ger","7-Cit","8-Van"]].mean(axis=0)
    
    pd.concat([p,s], axis=1).rename(columns={0:"precision", 1:"sensitivity"}).plot.bar()
    plt.show()
    

if __name__ == "__main__":
    print(main())
