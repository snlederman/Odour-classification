"""
scales the cleaned data and stores it as such
"""

# packages
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from cmd_parse import get_args

def main():
    """program skeleton"""
    
    args = get_args()
    
    # loading clean data
    if args["augment"]:
        features_file = os.path.join(PROJECT_DIR,"data", "splitted", "train", "augmented", "features.csv")
    else:
        features_file = os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv")
        
    # loading raw data
    features = pd.read_csv(features_file)
    
    # set index for scaling
    features.set_index("ID", inplace=True)
    
    # scaling features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)

    features_scaled.reset_index(inplace=True)
    
    # saving scaled features
    if args["augment"]:
        features_scaled_file = os.path.join(PROJECT_DIR,"data", "splitted", "train", "augmented", "scaled", "features.csv")
    else:
        features_scaled_file = os.path.join(PROJECT_DIR,"data", "cleaned", "scaled", "features.csv")
    
    features_scaled.to_csv(features_scaled_file, index=False)
    


if __name__ == "__main__":
    main()