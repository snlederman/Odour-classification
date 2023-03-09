"""
feature extraction using savgol filter
"""

# packages
import pandas as pd
from scipy.signal import savgol_filter

def derive(features):
    """
    extracts the first two derivitives of the signals
    """

    first_derivitive = pd.DataFrame(
        savgol_filter(features, deriv=1, window_length=3, polyorder=1, axis=0),
        columns=features.columns,
        index=features.index
    )

    second_derivitive = pd.DataFrame(
        savgol_filter(features, deriv=2, window_length=3, polyorder=2, axis=0),
        columns=features.columns,
        index=features.index
    )

    features = pd.concat([features, first_derivitive, second_derivitive], axis=1)

    return features
