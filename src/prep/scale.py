"""
scale the cleaned data and storing it as such
"""

# packages
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

def main():
    """program skeleton"""
    
    # loading raw data
    features_file = os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv")
    features = pd.read_csv(features_file)
    
    # set index for scaling
    features.set_index("ID", inplace=True)
    
    # scaling features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)

    features_scaled.reset_index(inplace=True)
    
    # saving scaled features
    features_scaled.to_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "scaled", "features.csv"), index=False)


if __name__ == "__main__":
    main()