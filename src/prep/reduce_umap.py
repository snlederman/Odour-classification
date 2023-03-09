"""
UMAP 

"""

# packages
import pandas as pd
import umap.umap_ as umap

def reduce(labels, features):
    """
    running linear discriminant analysis
    for supervised dimentionality reduction
    """
    reducer = umap.UMAP() 
    feature_umap = reducer.fit_transform(features)
    
    feature_umap = pd.DataFrame(
        reducer.fit_transform(features),
        columns=features.columns,
        index=features.index
    )

    return feature_umap



