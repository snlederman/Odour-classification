"""
ada boost model
"""

# packages
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

def ada_boost(x_train, y_train, x_test, y_test):
    """ada boost model"""
    model = AdaBoostClassifier()
    model.fit(x_train, y_train["label"])
    
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return report
