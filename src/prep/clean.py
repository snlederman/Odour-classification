"""
cleans the raw data and storing it as such
"""

# packages
import os
import pandas as pd

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

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
    features = features[sorted(features.columns)]

    # saving labels
    labels.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "labels.csv"), index=False)
    
    # saving features
    features.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv"), index=False)


if __name__ == "__main__":
    main()