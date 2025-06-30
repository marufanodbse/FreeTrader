import json
import talib
import pandas as pd
import numpy as np
import keras_tuner as kt

from common.util import get_current_kline_timestamp, parse_interval_to_microseconds
from finance.exchange import Exchange

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, LeakyReLU, BatchNormalization,
    Dropout, GlobalAveragePooling1D, Dense, Add, LayerNormalization
)
from tensorflow.keras.regularizers import l2

from models.lstm.model_utils import load_latest_model, save_model
from models.lstm.multioutput_early_stopping import MultiOutputEarlyStopping

import optuna
from optuna.pruners import HyperbandPruner
from sklearn.metrics import mean_squared_error

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)  # 监控总验证损失

# early_stopping = EarlyStopping(
#     monitor="val_output_1_accuracy",  # 监控 validation 上 output_1 的准确率
#     patience=5,
#     mode="max",  # 因为 accuracy 越高越好
#     restore_best_weights=True,
# )


def build_tcn_model(
    seq_length, n_features, filters, kernel_size, dilations, dropout_rate, learning_rate, l2_reg
):  
    model = Sequential()
    model.add(Input(shape=(seq_length, n_features)))
    for i, dilation in enumerate(dilations):
        model.add(
            Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                padding="causal",
                dilation_rate=dilation,
                kernel_regularizer=l2(l2_reg),
            )
        )
        model.add(LayerNormalization())
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dropout(dropout_rate))

    model.add(Conv1D(filters=filters//2, kernel_size=1, activation="relu"))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(1))  # 输出一个价格值
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


class TCNModel:
    def __init__(self, args):
        self.args = args
        config = {}
        if "proxy" in args:
            config["proxies"] = {
                "http": args["proxy"],
                "https": args["proxy"],
            }
        self.exchange = Exchange(args.get("exchange", "binance"), config)

        self.symbol = args.get("symbol")
        self.timeframe = args.get("timeframe")
        self.args = self.args

        self.features = []
        self.model = None
        self.init = False

        # print(f"init args:{args}")

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

    def build_features(self, df):
        self.features.extend(["open", "high", "low", "close", "volume"])

        df = df.copy()
        df.loc[:, "rsi"] = talib.RSI(df["close"], timeperiod=14)
        self.features.append("rsi")
        df.loc[:, "macd"], df.loc[:, "macd_signal"], df.loc[:, "macd_hist"] = (
            talib.MACD(df["close"])
        )
        self.features.extend(["macd", "macd_signal", "macd_hist"])

        df.loc[:, "bb_upper"], df.loc[:, "bb_middle"], df.loc[:, "bb_lower"] = (
            talib.BBANDS(df["close"])
        )
        self.features.extend(["bb_upper", "bb_middle", "bb_lower"])

        # K线行为因子
        df.loc[:, "kline_range"] = df["high"] - df["low"]
        df.loc[:, "kline_body"] = (df["close"] - df["open"]).abs()
        df.loc[:, "upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
        df.loc[:, "lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
        self.features.extend(
            ["kline_range", "kline_body", "upper_shadow", "lower_shadow"]
        )

        df = df.dropna()
        return df

    def preprocess_data(self, data, seq_length=60):

        data = self.build_features(data)
        data = data[self.features]
        data["close"] = data["close"].rolling(window=3, min_periods=1).mean()
        data = data.fillna(method="ffill")

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        close_scaler = MinMaxScaler()
        close_scaler.fit(data[["close"]])

        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i : i + seq_length])
                y.append(data[i + seq_length, 3])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data, seq_length)
        train_size = int(len(X) * 0.70)
        val_size = int(len(X) * 0.12)
        X_train, X_val, X_test = (
            X[:train_size],
            X[train_size : train_size + val_size],
            X[train_size + val_size :],
        )
        y_train, y_val, y_test = (
            y[:train_size],
            y[train_size : train_size + val_size],
            y[train_size + val_size :],
        )

        return X_train, y_train, X_val, y_val, X_test, y_test, scaler, close_scaler

    def load_model(self):
        self.model, self.scaler, self.last_timestamp = load_latest_model(
            self.symbol, self.timeframe
        )
        if self.model & self.scaler:
            self.init = True

    def init_model(self, df):
        if df is None:
            df = self.fetch_new_data(
                int(datetime.now().timestamp() - 24 * 3600 * self.args.get("days"))
                * 1000
            )

        
        seq_length = self.args.get("seq_length", 60)

        X_train, y_train, X_val, y_val, X_test, y_test, scaler, close_scaler = (
            self.preprocess_data(df, seq_length)
        )

        self.model = build_tcn_model(
            seq_length=seq_length,
            n_features=len(self.features),
            filters=self.args.get("filters"),
            kernel_size=self.args.get("kernel_size"),
            dilations=[2**i for i in range(self.args.get("n_layers"))],
            dropout_rate=self.args.get("dropout_rate"),
            learning_rate=self.args.get("learning_rate"),
            l2_reg=self.args.get("l2_reg"),
        )

        # X_train = np.random.randn(100, 60, 16).astype(np.float32)
        # y_train = np.random.randn(100, 1).astype(np.float32)
        self.model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=self.args.get("batch_size"),
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=[early_stopping],
        )
        y_val_pred = self.model.predict(X_val, verbose=0)
        return y_val, y_val_pred

    def clean(self):
        del self.mode

    def optimize(self):

        df = self.fetch_new_data(
            int(datetime.now().timestamp() - 24 * 3600 * self.args.get("days")) * 1000
        )

        def objective(trial):
            self.args["seq_length"] = trial.suggest_categorical("seq_length", [30, 60, 90, 120])
            self.args["filters"] = trial.suggest_int("filters", 16, 64)
            self.args["kernel_size"] = trial.suggest_int("kernel_size", 2, 7)
            self.args["dropout_rate"] = trial.suggest_categorical("dropout_rate", [0.1, 0.2, 0.3, 0.4, 0.5])
            self.args["n_layers"] = trial.suggest_int("n_layers", 2, 6)
            self.args["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
            self.args["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            self.args["l2_reg"] = trial.suggest_float("l2_reg", 1e-5, 1e-3, log=True)
            # self.args["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            # self.args["activation"] = trial.suggest_categorical("activation", ["relu", "elu", "gelu"])

            y_val, y_val_pred = self.init_model(df)
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            return mean_squared_error(y_val, y_val_pred)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
        print(f"最佳超参数: {best_params}")
    
    def show_prediction_vs_actual(self, days, epochs=50):
        df = self.fetch_new_data(
            int(datetime.now().timestamp() - 24 * 3600 * days) * 1000
        )

        self.init_model(df)
        x = (
            pd.to_datetime(df_20["timestamp"], unit="ms")
            .dt.tz_localize("UTC")
            .dt.tz_convert("Asia/Shanghai")
            .iloc[self.window_size - len(df_20) :]
        )

        x_data, y_data_true = self.create_dataset(df_20)
        loss_value = self.model.evaluate(x_data, y_data_true, 1)

        # 模型预测（仍是归一化值）
        y_pred = self.model.predict(x_data)
        y_pred = y_pred.flatten()

        # 构造 dummy 矩阵来反归一化
        dummy_pred = np.zeros((len(y_pred), self.scaler.n_features_in_))
        dummy_true = np.zeros((len(y_data_true), self.scaler.n_features_in_))

        dummy_pred[:, 0] = y_pred
        dummy_true[:, 0] = y_data_true

        # 反归一化
        y_pred_real = self.scaler.inverse_transform(dummy_pred)[:, 0]
        y_true_real = self.scaler.inverse_transform(dummy_true)[:, 0]

        # 绘图

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_true_real,
                mode="lines",
                name="真实价格",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_pred_real,
                mode="lines",
                name="预测价格",
                line=dict(color="red", dash="dash"),
            )
        )

        fig.update_layout(
            title=f"LSTM Prediction vs True {loss_value}",
            xaxis_title="时间",
            yaxis_title="价格",
            legend=dict(x=0, y=1),
            template="plotly_white",
            autosize=True,
            height=500,
        )
        import webbrowser, os

        file_path = (
            f'datas/predictions/{self.symbol.replace("/", "-")}_{self.timeframe}.html'
        )
        if os.path.exists(file_path):
            os.remove(file_path)
        fig.write_html(file_path)

        webbrowser.open(f"file://{os.path.abspath(file_path)}")
