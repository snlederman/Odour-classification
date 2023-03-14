"""
Multi-layer Perceptron classifier.
"""

# packages
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

def MLP(x_train, y_train, x_test, y_test):
    """mlp model"""
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs')
    model.fit(x_train, y_train["label"])

    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, report
