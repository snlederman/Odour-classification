"""
dimentionality reduction
"""

# packages
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def reduce(labels, features):
    """
    running linear discriminant analysis
    for supervised dimentionality reduction
    """

    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
    lda.fit(features, labels["label"])
    features_lda = pd.DataFrame(
        lda.transform(features),
        index=features.index
    )

    return features_lda
