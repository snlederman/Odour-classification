"""
random forest model + recursive feature elimination.
"""

# packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report

def random_forest(x_train, y_train, x_test, y_test):
    """random forest model"""
    model = RandomForestClassifier()
    for num in range(30,35):
        selector = RFE(model, n_features_to_select=num, step=1)
        selector.fit(x_train, y_train["label"])
        x_train_reduced = x_train.loc[:,selector.support_]
        x_test_reduced = x_test.loc[:,selector.support_]
        
        model.fit(x_train_reduced, y_train["label"])
        y_pred = model.predict(x_test_reduced)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(report["accuracy"])

    return model, report
