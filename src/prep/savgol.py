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

def main():
    """program skeleton"""

    # loading data
    features_path = os.path.join(PROJECT_DIR, "data", "cleaned", "features.csv")
    features = pd.read_csv(features_path).set_index("ID")
    
    first_derivitive = pd.DataFrame(savgol_filter(features, deriv=1, window_length=3, polyorder=1, axis=0), columns=features.columns, index=features.index)
    second_derivitive = pd.DataFrame(savgol_filter(features, deriv=2, window_length=3, polyorder=2, axis=0), columns=features.columns, index=features.index)
    
    features = pd.concat([features, first_derivitive, second_derivitive], axis=1)
    
    features.reset_index(inplace=True)
    # saving data
    features.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "derived", "features.csv"), index=False)


if __name__ == "__main__":
    main()
