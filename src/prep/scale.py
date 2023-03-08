"""
scales the cleaned data and stores it as such
"""

# packages
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale(features):
    """
    scale the data using standard scaling
    """

    # scaling features
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )

    return features_scaled
