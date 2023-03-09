"""
splitting data for modeling
"""

# packages
import numpy as np

def split(labels, features):
    """
    splitting the dataset to train and test
    taking into account antenea ID to simulate true test with new anteneas
    """
    labels.reset_index(inplace=True)
    features.reset_index(inplace=True)
    
    unique_IDs = labels["ID"].unique()
    # split to 80% training (later 60-20 with validation) and 20% testing
    train_num = int(len(unique_IDs) * 0.8)
    train_IDs = np.random.choice(unique_IDs, train_num, replace=False)
    test_IDs = unique_IDs[~np.isin(unique_IDs, train_IDs)]

    y_train = labels.loc[np.isin(labels["ID"],train_IDs),:]
    x_train = features.loc[np.isin(features["ID"],train_IDs),:]

    y_test = labels.loc[np.isin(labels["ID"],test_IDs),:]
    x_test = features.loc[np.isin(features["ID"],test_IDs),:]

    for df in [x_train, y_train, x_test, y_test]:
        df.set_index("ID", inplace=True)
        
    return  x_train, y_train, x_test, y_test
