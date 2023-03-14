"""
recursive neural network
"""

# packages
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


def get_prediction(y, labels):
    return labels[y.argmax(axis=1)]
    
def RNN(x_train, y_train, x_test, y_test):
    """running recursive neural network model"""
    enc = OneHotEncoder()

    unique_labels = y_train["label"].unique()
    enc.fit(unique_labels.reshape(-1, 1))

    # Convert sparse tensor to dense tensor and cast to float32
    y_train_encoded = enc.transform(y_train["label"].values.reshape(-1, 1)).toarray().astype(float)
    y_test_encoded = enc.transform(y_test["label"].values.reshape(-1, 1)).toarray().astype(float)

    in_shape = x_train.shape[1]
    print(f"input shape: {in_shape}")
    hidden1 = int(in_shape / 2)
    print(f"LSTM hidden: {hidden1}")
    hidden2 = int(hidden1 / 2)
    print(f"dense hidden: {hidden2}")
    out_shape = len(unique_labels)
    print(f"output shape: {out_shape}")

    model = Sequential([
        layers.LSTM(hidden1, return_sequences=True, input_shape=(in_shape,1)),
        layers.LSTM(hidden1, return_sequences=True),
        layers.LSTM(hidden1, return_sequences=True),
        layers.LSTM(hidden1),
        layers.Dense(hidden2, activation="relu"),
        layers.Dense(hidden2, activation="relu"),
        layers.Dense(hidden2, activation="relu"),
        layers.Dense(out_shape, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(x_train.values, y_train_encoded, validation_data=(x_test.values, y_test_encoded), batch_size=10, epochs=100, callbacks=[callback])

    # plot_history(history, "loss", os.path.join(PROJECT_DIR, "docs", "figures", "training_history.png"))

    y_pred_encoded = model.predict(x_test)
    y_pred = get_prediction(y_pred_encoded, enc.categories_[0])
    y_true = get_prediction(y_test_encoded, enc.categories_[0])

    report = classification_report(y_true, y_pred, output_dict=True)

    return model, report
