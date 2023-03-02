"""
visualizing the data 
"""

# packages
import os
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

def main():
    """program skeleton"""
    features = pd.read_csv(os.path.join(PROJECT_DIR, "data", "cleaned", "features.csv"))
    labels = pd.read_csv(os.path.join(PROJECT_DIR, "data", "cleaned", "labels.csv"))
    
    features_augmented = pd.read_csv(os.path.join(PROJECT_DIR, "data", "augmented", "features.csv"))
    labels_augmented = pd.read_csv(os.path.join(PROJECT_DIR, "data", "augmented", "labels.csv"))
    
    for label in labels["label"].unique():
        label_indices = labels["label"] == label
        temp = features[label_indices].set_index("ID")
        
        labels_augmented_indices = labels_augmented["label"] == label
        temp_augmented = features_augmented[labels_augmented_indices].set_index("ID")
        
        temp.transpose().plot(legend=False)
        plt.title(f"Odor: {label}")
        plt.savefig(os.path.join(PROJECT_DIR, "docs", "figures", "augmented_signals", f"Odor_{label}.png"))
        
        temp_augmented.transpose().plot(legend=False)
        plt.title(f"Odor: {label}_augmented")
        plt.savefig(os.path.join(PROJECT_DIR, "docs", "figures", "augmented_signals", f"Odor_{label}_augmented.png"))


if __name__ == "__main__":
    main()
