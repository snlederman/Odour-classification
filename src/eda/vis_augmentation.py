"""
visualizing the data 
"""

# packages
import os
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

def main():
    """program skeleton"""
    features = pd.read_csv(os.path.join(PROJECT_DIR, "data", "splitted", "train", "features.csv"))
    labels = pd.read_csv(os.path.join(PROJECT_DIR, "data", "splitted", "train", "labels.csv"))
    
    features_augmented = pd.read_csv(os.path.join(PROJECT_DIR, "data", "splitted", "train", "augmented", "features.csv"))
    labels_augmented = pd.read_csv(os.path.join(PROJECT_DIR, "data", "splitted", "train", "augmented", "labels.csv"))
    
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
