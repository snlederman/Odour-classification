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
    data_file = os.path.join(PROJECT_DIR,"data", "raw", "8_odors_new_names.csv")
    data = pd.read_csv(data_file)

    # removing unnecessary columns
    data.drop(columns=["date", "channel"], inplace=True)

    labels = data["label"]
    features = data.drop(columns=["label"])

    # scaling features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features))

    # saving data
    labels.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "labels.csv"), index=False)
    features_scaled.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv"), index=False)


if __name__ == "__main__":
    main()