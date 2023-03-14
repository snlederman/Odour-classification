"""
gradient boosting classifier model (defaut parameter)
"""

# packages
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

def gradient_boosting(x_train, y_train, x_test, y_test):
    """gradient boosting classifier model"""
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train["label"])

    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report
