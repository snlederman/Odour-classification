"""
visualizing the data 
"""

# packages
import os
import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

def main():
    """program skeleton"""

    data = pd.read_csv(os.path.join(PROJECT_DIR, "data", "raw", "single_odor.csv"))
    data.drop(columns=["date","channel"], inplace=True)
    
    data.set_index("ID", inplace=True)
    
    feature_names = data.drop(columns=["label"]).columns
    
    anova_dict = {"time" : [], "pvalue" : []}
    for time_point in feature_names:
        current_time_point = data[time_point]
        time_point_groups = []
        for antennea in data.index.unique():
            time_point_groups.append(current_time_point[current_time_point.index == antennea])
        
        current_pvalue = f_oneway(*time_point_groups).pvalue
        
        anova_dict["pvalue"].append(current_pvalue)
        anova_dict["time"].append(time_point)
    
    anova_df = pd.DataFrame(anova_dict)
    anova_df["log_inverse_pvalue"] = np.log((1 / anova_df["pvalue"]))
    anova_df.to_csv(os.path.join(PROJECT_DIR, "data", "ANOVA_per_time_point.csv"), index=False)
    anova_df["log_inverse_pvalue"].plot()
    plt.xlabel("time")
    plt.ylabel("log inverse p-value")
    plt.title("ANOVA per time point")
    plt.savefig(os.path.join(PROJECT_DIR, "docs", "figures", "ANOVA_per_time_point.png"))



if __name__ == "__main__":
    main()
