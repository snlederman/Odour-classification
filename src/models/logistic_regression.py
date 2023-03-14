"""
Logistic regression model (defaut parameter)
"""

# packages
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def logistic_regression(x_train, y_train, x_test, y_test):
    """logistic regression model"""
    model = LogisticRegression()
    model.fit(x_train, y_train["label"])
    
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report
