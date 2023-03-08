"""
dimentionality reduction
"""

# packages
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def reduce(labels, features):
    """
    running linear discriminant analysis
    for supervised dimentionality reduction
    """

    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    lda.fit(features, np.ravel(labels))
    features_lda = lda.transform(features)

    return features_lda
