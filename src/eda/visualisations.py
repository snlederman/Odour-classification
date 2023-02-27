"""
visualizing the data 
"""

# packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

def main():
    """program skeleton"""
    features = pd.read_csv(os.path.join(PROJECT_DIR, "data", "cleaned", "features.csv"))
    labels = pd.read_csv(os.path.join(PROJECT_DIR, "data", "cleaned", "labels.csv"))
    
    data = pd.merge(labels, features)
    data.set_index("ID", inplace=True)
    
    feature_names = features.drop(columns=["ID"]).columns
    num_features = len(feature_names)
    data.rename(columns={ftr:int(ftr.replace('t','')) for ftr in feature_names}, inplace=True)
    
    for label in data["label"].unique():
        for antennea in data.index.unique():
            df_slice = (data["label"] == label).values & (data.index == antennea)
            temp_df = data.loc[df_slice, list(range(num_features))]
            if len(temp_df) > 0:
                temp_df.transpose().plot(legend=False)
                plt.title(f"antennea id: {antennea}    odor: {label}")
                plt.savefig(os.path.join(PROJECT_DIR, "docs", "figures", f"antennea_{antennea}_{label}.png"))

    
if __name__ == "__main__":
    main()
