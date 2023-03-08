"""
clipes the signal
"""

def clip(features, start=10, end=111):
    """
    clips the features to smaller time frame
    """
    features_clipped = features.iloc[:,start:end]
    return features_clipped
