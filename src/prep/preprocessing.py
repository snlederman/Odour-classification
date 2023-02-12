"""
preprocessing the raw data and storing it as such
"""

# packages
# import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

def main():
    """program skeleton"""
    data_file = os.path.join(PROJECT_DIR,"data", "raw", "single_odor.csv")

    data = pd.read_csv(data_file)

    # removing unnecessary columns
    data.drop(columns=["date", "channel"], inplace=True)

    labels = data[["label","ID"]].set_index("ID")
    features = data.drop(columns=["label"]).set_index("ID")

    # scaling features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)

    # reset index for saving
    labels.reset_index(inplace=True)
    features_scaled.reset_index(inplace=True)

    # saving data
    labels.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "labels.csv"), index=False)
    features_scaled.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv"), index=False)


if __name__ == "__main__":
    main()