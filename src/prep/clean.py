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
    data_file = os.path.join(PROJECT_DIR,"data", "raw", "single_odor.csv")
    data = pd.read_csv(data_file)

    # removing unnecessary columns
    data.drop(columns=["date", "channel"], inplace=True)

    # splitting to features and labels
    labels = data[["ID","label"]]
    features = data.drop(columns=["label"])

    # saving labels
    labels.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "labels.csv"), index=False)
    
    # saving features
    features.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv"), index=False)


if __name__ == "__main__":
    main()