"""
augments the data and stores it as such
"""

# packages
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from cmd_parse import get_args

def augment(df):
    """
    doing a random weighted average
    
    takes in an array and generates random weights in the same size
    calculates weighted average + some noise
    returns a single sample
    """
    nrow = df.shape[0]
    weights = np.random.randint(1,100,nrow)
    noise = 0.005
    noise_range = (1-noise, 1+noise)
    
    sample = df.apply(lambda col: col * weights * np.random.uniform(*noise_range)).sum(axis=0) / weights.sum()
    sample_df = pd.DataFrame(sample).transpose()
    return sample_df

def main():
    """program skeleton"""
    args = get_args()
    
    # loading cleaned data
    if args["scale"]:
        features_file = os.path.join(PROJECT_DIR,"data", "cleaned", "scaled", "features.csv")
    else:
        features_file = os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv")
        
    features = pd.read_csv(features_file)
    
    labels_file = os.path.join(PROJECT_DIR,"data", "cleaned", "labels.csv")
    labels = pd.read_csv(labels_file)
    
    # set id index
    features.set_index("ID", inplace=True)

    for n in range(args["number"]):
        for label in labels["label"].unique():
            new_index = labels["ID"].max() + 1
            label_indice = labels["label"] == label
            label_features = features[label_indice]
            sample_features = augment(label_features.sample(2))
            sample_label = pd.DataFrame({"ID":new_index, "label":label}, index=[0])
            sample_features.index = [new_index]

            # append new row
            labels = pd.concat([labels, sample_label], axis=0, ignore_index=True)
            features = pd.concat([features, sample_features], axis=0)

    labels.to_csv(os.path.join(PROJECT_DIR,"data", "augmented", "labels.csv"), index=False)
    
    features.reset_index(inplace=True)
    features.rename(columns={"index":"ID"}, inplace=True)
    
    if args["scale"]:
        features.to_csv(os.path.join(PROJECT_DIR,"data", "augmented", "scaled", "features.csv"), index=False)
    else:
        features.to_csv(os.path.join(PROJECT_DIR,"data", "augmented", "features.csv"), index=False)
        

if __name__ == "__main__":
    main()