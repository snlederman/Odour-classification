"""
base line model, predicts most common label
"""

# packages
import numpy as np
from sklearn.metrics import classification_report


class MostCommon:
    """predicting alwase the most common label in training data"""
    def __init__(self):
        self.labels = None
        
    def fit(self, X, y):
        """recives pandas series as training labels and store them for predictions"""
        most_common_label = y.value_counts().index[0]
        self.labels = most_common_label

    def predict(self, X):
        """predict random labels from training set at the length of input"""
        res = np.array([self.labels] * X.shape[0]).astype(object)
        return res


def most_common(x_train, y_train, x_test, y_test):
    """
    predicts most common label
    """
    model = MostCommon()
    model.fit(x_train, y_train["label"])
    y_pred = model.predict(x_test)

    report = classification_report(y_test["label"], y_pred, output_dict=True)
    return report
