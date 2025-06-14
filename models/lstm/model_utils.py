import tensorflow as tf
import joblib
import os
import json
import numpy as np
from datetime import datetime

MODEL_DIR = "datas/models"

def create_dataset(data, target_col_idx, window_size):
    x, y = [], []
    for i in range(window_size, len(data)):
        x.append(data[i - window_size : i, :-1])
        y.append(data[i, target_col_idx])
    return np.array(x), np.array(y)


def create_model(input_shape, num_units, learning_rate, dropout):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(num_units, return_sequences=True),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.LSTM(num_units),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
    )
    return model


def save_model(model, scaler, symbol, timeframe, timestamp):
    os.makedirs(MODEL_DIR, exist_ok=True)
    key = f'{symbol.replace("/", "-")}_{timeframe}'
    model.save(f"{MODEL_DIR}/lstm_{key}_{timestamp}.h5")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler_{key}_{timestamp}.save")


def load_latest_model(symbol, timeframe):
    os.makedirs(MODEL_DIR, exist_ok=True)
    key = f'{symbol.replace("/", "-")}_{timeframe}'
    models = sorted(f for f in os.listdir(MODEL_DIR) if f.startswith(f"lstm_{key}"))
    if not models:
        return None, None, None
    latest = models[-1]
    timestamp = latest.split("_")[3].replace(".h5", "")

    model = tf.keras.models.load_model(f"{MODEL_DIR}/lstm_{key}_{timestamp}.h5")
    scaler = joblib.load(f"{MODEL_DIR}/scaler_{key}_{timestamp}.save")
    return model, scaler, int(timestamp)
