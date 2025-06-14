import pandas as pd
import numpy as np
import talib
import tensorflow as tf
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from common.util import parse_interval_to_microseconds
from finance.exchange import Exchange
from models.model import BaseLSTMModel
from models.model_utils import create_dataset, create_model


class PriceLSTMMode(BaseLSTMModel):

    def __init__(self, args):
        super.__init__(args)

    def evaluate(self, df, verbose=1):
        x_val, y_val = self.create_dataset(df)
        return self.model.evaluate(x_val, y_val, verbose)

    def build_features(self, df):
        df = df.copy()
        df.loc[:, "rsi"] = talib.RSI(df["close"], timeperiod=14)
        df.loc[:, "stoch_k"], df.loc[:, "stoch_d"] = talib.STOCH(
            df["high"], df["low"], df["close"]
        )
        df.loc[:, "macd"], df.loc[:, "macd_signal"], df.loc[:, "macd_hist"] = (
            talib.MACD(df["close"])
        )
        df.loc[:, "ema_5"] = talib.EMA(df["close"], timeperiod=5)
        # # merged_df['ema_10'] = talib.EMA(merged_df['close'], timeperiod=10)
        df.loc[:, "ema_20"] = talib.EMA(df["close"], timeperiod=20)
        # merged_df['adx'] = talib.ADX(merged_df['high'], merged_df['low'], merged_df['close'])
        # merged_df['bb_upper'], merged_df['bb_middle'], merged_df['bb_lower'] = talib.BBANDS(merged_df['close'])
        df.loc[:, "obv"] = talib.OBV(df["close"], df["volume"])
        # merged_df["volume_ratio"] = merged_df["volume"] / merged_df["volume"].rolling(5).mean()

        # K线行为因子
        df.loc[:, "kline_range"] = df["high"] - df["low"]
        df.loc[:, "kline_body"] = (df["close"] - df["open"]).abs()
        df.loc[:, "upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
        df.loc[:, "lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
        df = df[[col for col in df.columns if col != "timestamp"] + ["timestamp"]]
        df = df.dropna()
        return df

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

    def show_prediction_vs_actual(self, days, epochs=50):
        df = self.fetch_new_data(
            int(datetime.now().timestamp() - 24 * 3600 * days) * 1000
        )
        df_20 = df[len(df) - int(len(df) * 0.8) :]
        df_80 = df[0 : int(len(df) * 0.8)]
        self.init_model(epochs=epochs, df=df_80)
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
