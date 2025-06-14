from abc import abstractmethod
import pandas as pd
import numpy as np
import talib
import tensorflow as tf
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from common.util import parse_interval_to_microseconds
from finance.exchange import Exchange
from models.lstm.model_utils import create_dataset

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

class BaseLSTMModel:
    def __init__(self, args):
        self.args = args
        config = {}
        if "proxy" in args:
            config["proxies"] = {
                "http": args["proxy"],
                "https": args["proxy"],
            }
        self.exchange = Exchange(args.get("exchange", "binance"), config)

        self.offsetSteps = args.get("offsetSteps", [1, 3, 6, 12])
        self.feature_cols = [
            "close",
            "open",
            "high",
            "low",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "kline_range",
            "kline_body",
            "upper_shadow",
            "lower_shadow",
        ]

        self.symbol = args.get("symbol")
        self.timeframe = args.get("timeframe")
        self.batch_size = args.get("batch_size", 128)
        self.seq_len = args.get("seq_len", 120)
        self.units = args.get("num_units", 96)
        self.dropout = args.get("dropout", 0.2)
        self.learning_rate = args.get("learning_rate", 1e-3)
        self.scaler = None
        self.model = None
        self.init = False
        print(f"init TrendLstm args:{args}")

    @abstractmethod
    def create_model(self, num_units, learning_rate, dropout):
        pass

    @abstractmethod
    def target_col_idx(self):
        pass

    def fetch_new_data(self, since, count=10000000):
        ohlcvs = self.exchange.fetch_ohlcv_batch(
            self.symbol, self.timeframe, since, count
        )
        if ohlcvs[-1][
            0
        ] > datetime.now().timestamp() * 1000 - parse_interval_to_microseconds(
            self.timeframe
        ):
            ohlcvs = ohlcvs[:-1]
        return pd.DataFrame(
            ohlcvs, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    def create_dataset(self, df):
        if self.scaler is None:
            self.scaler = MinMaxScaler()
        features = self.build_features(df)
        scaled_data = self.scaler.fit_transform(features)
        return create_dataset(scaled_data, self.target_col_idx(), self.window_size)

    def init_model(self, df, epochs=50):
        self.scaler = MinMaxScaler()
        x_data, y_data = self.create_dataset(df)

        split_idx = int(len(x_data) * 0.8)
        x_train, x_val = x_data[:split_idx], x_data[split_idx:]
        y_train, y_val = y_data[:split_idx], y_data[split_idx:]

        self.model = self.create_model(
            (x_train.shape[1], x_train.shape[2]),
            self.num_units,
            self.learning_rate,
            self.dropout,
        )

        self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            shuffle=False,
        )

        x_all = np.concatenate([x_train, x_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)

        if isinstance(self.model.optimizer.learning_rate, tf.Variable):
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, 1e-4)
        else:
            self.model.optimizer.learning_rate = 1e-4

        self.model.fit(
            x_all, y_all, epochs=5, batch_size=self.batch_size, shuffle=False, verbose=1
        )

    def train(self):
        df = self.fetch_new_data(
            self.last_timestamp
            - self.window_size * parse_interval_to_microseconds(self.timeframe)
        )
        x, y = self.create_dataset(df)
        if len(x) > 0:
            loss = self.model.train_on_batch(x, y)
            print(f"[{datetime.now()}] Incremental Loss: {loss:.6f}")
        else:
            print("⚠️ 数据不足，无法训练")

        self.df_cache = pd.concat([self.df_cache, df])
        self.last_timestamp = df["timestamp"].iloc[-1] + parse_interval_to_microseconds(
            self.timeframe
        )
