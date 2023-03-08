"""

uses the fourier transform and stores it as such
"""

# packages
import os
import sys
import pandas as pd
from scipy.fft import fft, ifft
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
    
    features_file = os.path.join(PROJECT_DIR,"data", "cleaned", "features.csv")
        
    # loading raw data
    features = pd.read_csv(features_file)
    
    # set index for scaling
    features.set_index("ID", inplace=True)
    
    # fft on the signals
    # X_train_fft = X_train.apply(lambda row: fft(row.to_numpy()), axis=1)
    
    features_fft = features.apply(lambda row: fft(row.to_numpy()), axis=1, result_type='expand')
    
    features_fft_file = os.path.join(PROJECT_DIR,"data", "cleaned", "fourier", "features.csv")
    
    features_fft.reset_index(inplace=True)
    features_fft.to_csv(features_fft_file, index=False)
    


if __name__ == "__main__":
    main()