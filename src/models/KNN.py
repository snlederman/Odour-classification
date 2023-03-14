"""
clustering KNN model
"""

# packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def KNN(x_train, y_train, x_test, y_test):
    """KNN model"""
    n_odors = len(y_train["label"].unique())
    model = KNeighborsClassifier(n_neighbors = n_odors)
    model.fit(x_train, y_train["label"])
    
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, report
