"""
uses the fourier transform
"""

# packages
from scipy.fft import fft

def fourier(features):
    """
    transforms feature using fourier transformation
    """

    features_fft = features.apply(lambda row: fft(row.to_numpy()), axis=1, result_type='expand')
    return features_fft