"""
splitting data for modeling
"""

# packages
# import numpy as np
import pandas as pd
import os

SCRIPT_PATH = os.path.realpath(__file__)
PROJECT_DIR = SCRIPT_PATH.split("src")[0]

def main():
    """program skeleton"""
    labels = pd.read_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "labels.csv"))
    features = pd.read_csv(os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv"))



if __name__ == "__main__":
    main()