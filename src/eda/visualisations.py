"""
visualizing the data 
"""

# packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

def main():
    """program skeleton"""
    # features = pd.read_csv(os.path.join(PROJECT_DIR, "data", "cleaned", "features.csv"))
    # labels = pd.read_csv(os.path.join(PROJECT_DIR, "data", "cleaned", "labels.csv"))
    
    data = pd.read_csv(os.path.join(PROJECT_DIR, "data", "raw", "single_odor.csv"))
    data.drop(columns=["date","channel"], inplace=True)
    
    # data = pd.merge(labels, features)
    data.set_index("ID", inplace=True)
    
    feature_names = data.drop(columns=["label"]).columns
    label_names_replace = {label:label.split("-")[1] for label in data["label"].unique()}
    data["label"].replace(label_names_replace, inplace=True)
    
    # feature_names = features.drop(columns=["ID"]).columns
    num_features = len(feature_names)
    data.rename(columns={ftr:int(ftr.replace('t','')) for ftr in feature_names}, inplace=True)
    
    # plot example of signal from each odor
    samples_df = pd.DataFrame()
    for odor in label_names_replace.values():
        odor_df = data[data["label"] == odor]
        samples_df = pd.concat([samples_df, odor_df.sample()])
    
    to_plot = (
        samples_df
        .rename(columns={"label":"Odors"})
        .set_index("Odors")
        .transpose()
    )
    to_plot.plot()
    plt.xlabel("Time (seconds)")
    plt.title("Example of signal for different odors")
    plt.savefig(os.path.join(PROJECT_DIR, "docs", "figures", "example_signals.png"))
    
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
