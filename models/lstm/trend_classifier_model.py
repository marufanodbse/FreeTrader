import json
import talib
import pandas as pd
import numpy as np
import keras_tuner as kt

from common.util import get_current_kline_timestamp, parse_interval_to_microseconds
from finance.exchange import Exchange

from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from models.lstm.model_utils import load_latest_model, save_model
from models.lstm.multioutput_early_stopping import MultiOutputEarlyStopping

# early_stopping = EarlyStopping(
#     monitor="val_loss", patience=10, restore_best_weights=True
# )  # 监控总验证损失

early_stopping = EarlyStopping(
    monitor="val_output_1_accuracy",  # 监控 validation 上 output_1 的准确率
    patience=5,
    mode="max",  # 因为 accuracy 越高越好
    restore_best_weights=True,
)


def build_multi_output_lstm(
    input_shape, offsetSteps, units=64, dropout=0.2, learning_rate=0.001
):
    input_layer = Input(shape=input_shape)
    # 共享的 LSTM 层
    lstm_layer_1 = LSTM(units, return_sequences=False)(input_layer)
    dropout_lstm_1 = Dropout(dropout)(lstm_layer_1)

    # lstm_layer_2 = LSTM(units=int(units / 2), return_sequences=False)(dropout_lstm_1)
    # dropout_lstm_2 = Dropout(dropout)(lstm_layer_2)

    # 共享的全连接层 (用于进一步提取高层特征)
    shared_dense = Dense(int(units / 2), activation="relu")(dropout_lstm_1)
    dropout_shared = Dropout(dropout)(shared_dense)

    outputs = []
    for step in offsetSteps:
        # 在共享层之后，为每个输出添加一个或多个特定的小型 Dense 层
        specific_dense_1 = Dense(
            int(units / 4), activation="relu", name=f"specific_dense_1_{step}"
        )(dropout_shared)
        specific_dropout = Dropout(dropout)(specific_dense_1)

        out = Dense(3, activation="softmax", name=f"output_{step}")(specific_dropout)
        outputs.append(out)

    model = Model(inputs=input_layer, outputs=outputs)
    loss_funcs = {f"output_{step}": "categorical_crossentropy" for step in offsetSteps}

    # loss_weights = {
    #     f"output_{step}": 1 for step in self.offsetSteps
    # }

    metrics = {f"output_{step}": ["accuracy"] for step in offsetSteps}
    # print(metrics)
    model.compile(optimizer=Adam(learning_rate), loss=loss_funcs, metrics=metrics)
    # print(model.metrics_names)
    return model


class TrendClassifierLstm:
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

    def label_trend(self, df, offset, up_th=0.002, down_th=-0.002):
        future_close = df["close"].shift(-offset)
        pct_change = (future_close - df["close"]) / df["close"]

        df.loc[:, f"trend_{offset}"] = np.where(
            pct_change > up_th, 1, np.where(pct_change < down_th, -1, 0)
        )
        return df

    def build_features(self, df):
        df = df.copy()
        df.loc[:, "rsi"] = talib.RSI(df["close"], timeperiod=14)
        # df.loc[:, "stoch_k"], df.loc[:, "stoch_d"] = talib.STOCH(
        #     df["high"], df["low"], df["close"]
        # )
        df.loc[:, "macd"], df.loc[:, "macd_signal"], df.loc[:, "macd_hist"] = (
            talib.MACD(df["close"])
        )
        # df.loc[:, "ema_5"] = talib.EMA(df["close"], timeperiod=5)
        # df.loc[:, 'ema_10'] = talib.EMA(df['close'], timeperiod=10)
        # df.loc[:, "ema_20"] = talib.EMA(df["close"], timeperiod=20)
        # df.loc[:, 'adx'] = talib.ADX(df['high'], df['low'], df['close'])
        df.loc[:, "bb_upper"], df.loc[:, "bb_middle"], df.loc[:, "bb_lower"] = (
            talib.BBANDS(df["close"])
        )
        # df.loc[:, "obv"] = talib.OBV(df["close"], df["volume"])
        # merged_df["volume_ratio"] = merged_df["volume"] / merged_df["volume"].rolling(5).mean()

        # K线行为因子
        df.loc[:, "kline_range"] = df["high"] - df["low"]
        df.loc[:, "kline_body"] = (df["close"] - df["open"]).abs()
        df.loc[:, "upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
        df.loc[:, "lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]

        for step in self.offsetSteps:
            self.label_trend(df, step)

        df = df.dropna()
        return df

    def preprocess_data(self, df):
        # 特征选择
        if self.scaler is None:
            self.scaler = MinMaxScaler()

        features_scaled = self.scaler.fit_transform(df[self.feature_cols].values)

        # 构建序列样本和多标签
        X, y_list = [], {step: [] for step in self.offsetSteps}
        for i in range(self.seq_len, len(features_scaled)):
            X.append(features_scaled[i - self.seq_len : i])
            for step in self.offsetSteps:
                y_list[step].append(df[f"trend_{step}"].iloc[i])

        X = np.array(X)
        y_dict = {}
        for step in self.offsetSteps:
            y = np.array(y_list[step])
            y_dict[f"output_{step}"] = to_categorical(
                y + 1, num_classes=3
            )  # -1 => 0, 0 => 1, 1 => 2
        return X, y_dict

    def prepare_data(self, df, ratio=0.8):
        X, y_dict = self.preprocess_data(self.build_features(df))

        X = X[: -max(self.offsetSteps)]
        for step in self.offsetSteps:
            y_dict[f"output_{step}"] = y_dict[f"output_{step}"][
                : -max(self.offsetSteps)
            ]

        split_idx = int(len(X) * ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]

        y_train_dict = {}
        y_test_dict = {}

        for key in y_dict:
            y = y_dict[key]
            y_train_dict[key] = y[:split_idx]
            y_test_dict[key] = y[split_idx:]
        return X_train, X_test, y_train_dict, y_test_dict

    def init_model(self, df=None, epochs=30, ratio=0.8):
        if df is None:
            df = self.fetch_new_data(
                int(datetime.now().timestamp() - 24 * 3600 * self.args.get("days"))
                * 1000
            )
        self.last_timestamp = df["timestamp"].iloc[-1] + parse_interval_to_microseconds(
            self.timeframe
        )

        X_train, X_test, y_train_dict, y_test_dict = self.prepare_data(df, ratio)
        model = build_multi_output_lstm(
            X_train.shape[1:],
            self.offsetSteps,
            self.units,
            self.dropout,
            self.learning_rate,
        )

        early_stopping = MultiOutputEarlyStopping(
            monitors=[f"val_output_{step}_accuracy" for step in self.offsetSteps],
            patience=5,
            min_delta=0.001,
            mode="max",
            verbose=1,
            restore_best_weights=True,
        )

        history = model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_test, y_test_dict),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=[early_stopping],
            verbose=1,
        )

        val_accs = [
            history.history[f"val_output_{step}_accuracy"][-1]
            for step in self.offsetSteps
        ]

        self.model = model
        self.init = True

        # save_model(self.model, self.scaler, self.symbol, self.timeframe, int(datetime.now().timestamp()))

        return min(val_accs)

    def load_model(self):
        self.model, self.scaler, self.last_timestamp = load_latest_model(
            self.symbol, self.timeframe
        )
        if self.model & self.scaler:
            self.init = True

    def train_on_batch(self):
        if self.model is None | self.scaler is None:
            return
        df = self.fetch_new_data(
            self.last_timestamp - 1000 * parse_interval_to_microseconds(self.timeframe)
        )
        df = self.build_features(df)

        X, y_dict = self.preprocess_data(
            df[
                df["timestamp"]
                > (
                    self.last_timestamp
                    - self.seq_len * parse_interval_to_microseconds(self.timeframe)
                )
            ]
        )
        if len(X) > 0:
            self.model.train_on_batch(X, y_dict)
        else:
            print("⚠️ 数据不足，无法训练")

        self.df_cache = pd.concat([self.df_cache, df])
        self.last_timestamp = df["timestamp"].iloc[-1] + parse_interval_to_microseconds(
            self.timeframe
        )

    def predict(self):
        if self.init == False:
            raise RuntimeError("模型尚未训练，无法预测")
        timestamp = get_current_kline_timestamp(self.timeframe)
        df = self.fetch_new_data(
            timestamp - 500 * parse_interval_to_microseconds(self.timeframe)
        )
        print(f'predict: timestamp: {df["timestamp"].iloc[-1]}')

        X_input, _ = self.preprocess_data(self.build_features(df))
        last_seq = np.expand_dims(X_input[-1][-self.seq_len :], axis=0)
        predictions = self.model.predict(last_seq, verbose=1)

        print(predictions)
        results = {"timestamp": timestamp}
        for i, step in enumerate(self.offsetSteps):
            prob = predictions[i]
            results[f"output_{step}"] = (
                int(np.argmax(prob, axis=-1)) - 1,
                float(np.max(prob)),
            )  # -1 表示恢复原始标签（-1, 0, 1）
        return results

    # def optimize(self, df):
    #     X_train, X_test, y_train_dict, y_test_dict = self.prepare_data(df)

    #     print("optimize start run....")

    #     def build_model_fn(input_shape, offsetSteps):
    #         def build_model(hp):
    #             units = hp.Int("units", 64, 128, step=32)
    #             dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
    #             learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
    #             return build_multi_output_lstm(
    #                 input_shape, offsetSteps, units, dropout, learning_rate
    #             )

    #         return build_model

    #     tuner = kt.Hyperband(
    #         build_model_fn(X_train.shape[1:], self.offsetSteps),
    #         objective=kt.Objective("val_output_1_accuracy", direction="max"),
    #         max_epochs=30,
    #         factor=3,
    #         directory=f"datas/optimizes/",
    #         project_name=f"{self.symbol.replace('/', "-")}_{self.timeframe}",
    #     )

    #     tuner.search(
    #         X_train,
    #         y_train_dict,
    #         epochs=30,
    #         validation_data=(X_test, y_test_dict),
    #         callbacks=[early_stopping],
    #         verbose=1,
    #         executions_per_trial=1,
    #     )

    #     # 打印最优模型
    #     best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    #     print("✅ 最优超参数组合：")
    #     print(f"LSTM units: {best_hp.get('units')}")
    #     print(f"Dropout: {best_hp.get('dropout')}")
    #     print(f"Learning rate: {best_hp.get('learning_rate')}")
