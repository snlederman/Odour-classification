"""
preprocessing the raw data and storing it as such
"""

# packages
import os
import argparse
# import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

# argparse function
def get_args(argv=None):
    """
    Takes arguments from the user when running as a command line script
    """
    parser = argparse.ArgumentParser(description="Preproccess the raw data and saves it")
    parser.add_argument("-s","--scale", action="store_true", help="standard scaling")
    return vars(parser.parse_args(argv))

def main():
    """program skeleton"""
    args = get_args()
    
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
    
    if args["scale"]:
        # set index for scaling
        features.set_index("ID")
        
        # scaling features
        scaler = StandardScaler()
        features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)

        # saving scaled features
        features_scaled.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "scaled", "features.csv"), index=False)
    else:
        # saving features
        features.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv"), index=False)


if __name__ == "__main__":
    main()