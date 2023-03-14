"""
base line model to compare any new model to
"""

# packages
import numpy as np
from sklearn.metrics import classification_report


class RandomSampler:
    """random sampler from train labels"""
    def __init__(self):
        self.labels = None
        
    def fit(self, X, y):
        """recives pandas series as training labels and store them for predictions"""
        self.labels = y.values

    def predict(self, X):
        """predict random labels from training set at the length of input"""
        res = np.random.choice(self.labels, X.shape[0])
        return res


def random_sampler(x_train, y_train, x_test, y_test):
    """
    predicts random labels based on the train distribiution
    """

    model = RandomSampler()
    model.fit(x_train, y_train["label"])
    y_pred = model.predict(x_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    return model, report
