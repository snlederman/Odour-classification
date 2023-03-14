"""
random forest model
"""

# packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def random_forest(x_train, y_train, x_test, y_test):
    """random forest model"""
    model = RandomForestClassifier()
    model.fit(x_train, y_train["label"])

    y_pred = model.predict(x_test)
    report = classification_report(y_test["label"], y_pred, output_dict=True)

    return model, report
