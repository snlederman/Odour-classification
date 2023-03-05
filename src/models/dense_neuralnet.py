"""
random forest model
"""

# packages
import os
import sys
# import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

SCRIPT_PATH = os.path.realpath(__file__)

def get_project_dir(script_path):
    project_dir = script_path[:-script_path[::-1].find("crs")-3]
    return project_dir

PROJECT_DIR = get_project_dir(SCRIPT_PATH)

sys.path.append(os.path.join(PROJECT_DIR, "src", "utils"))
from cmd_parse import get_args
from load_data import load_data
from summeries_classification import summeries_multiclass_report
from log_classification import log_metrics
from training_history import plot_history

def get_prediction(y_pred):
    return y_pred.argmax(axis=1)
    
def main():
    """program skeleton"""
    args = get_args()
    
    X_train, y_train, X_test, y_test = load_data(PROJECT_DIR, args["scale"], args["augment"])
    
    unique_labels = y_train["label"].unique()
    label_encoder = dict(zip(unique_labels, range(len(unique_labels))))
    
    y_train_encoded = y_train.replace(label_encoder)
    y_test_encoded = y_test.replace(label_encoder)
    
    in_shape = X_train.shape[1]
    print(f"input shape: {in_shape}")
    hidden1 = int(in_shape / 2)
    print(f"first hidden: {hidden1}")
    hidden2 = int(hidden1 / 2)
    print(f"second hidden: {hidden2}")
    out_shape = len(unique_labels)
    print(f"output shape: {out_shape}")
    
    model = Sequential([
        Dense(hidden1, activation="relu", input_shape=(in_shape,)),
        Dense(hidden2, activation="relu"),
        Dense(out_shape, activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="mse", metrics="accuracy")

    callback = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    
    history = model.fit(X_train, y_train_encoded["label"], validation_data=(X_test, y_test_encoded["label"]), batch_size=2, epochs=20, callbacks=[callback])

    plot_history(history, "loss", os.path.join(PROJECT_DIR, "docs", "figures", "training_history.png"))
    
    y_pred_encoded = get_prediction(model.predict(X_test))
    
    label_decoder = {v: k for k, v in label_encoder.items()}
    y_pred = pd.Series(y_pred_encoded).replace(label_decoder)

    report = classification_report(y_test, y_pred, output_dict=True)
    log_metrics(report, model, PROJECT_DIR, args["scale"], args["augment"])
    report_summary = summeries_multiclass_report(report)
    
    return report_summary


if __name__ == "__main__":
    print(main())
