"""
cleans the raw data and stores it as such
"""

# packages
import os
import pandas as pd

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

def main():
    """program skeleton"""
    
    # loading raw data
    features_path = os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv")
    
    features = pd.read_csv(features_path).set_index(["ID"])
    features_clipped = features.iloc[:,10:111]
    features_clipped.reset_index(inplace=True)

    # saving features
    features_clipped.to_csv(os.path.join(PROJECT_DIR,"data", "clipped", "features.csv"), index=False)


if __name__ == "__main__":
    main()