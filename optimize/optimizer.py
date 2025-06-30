import os
import subprocess
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from common.util import format_timestamp, parse_interval_to_microseconds
from finance.exchange import Exchange

from models.lstm.trend_classifier_model import TrendClassifierLstm
from models.tcn.model import TCNModel

import optuna
from optuna.pruners import HyperbandPruner
from sklearn.metrics import mean_squared_error
# from models.price_model import BaseLSTMModel
# from models.model_utils import create_dataset, create_model

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)


def optimize(args):
    print("TensorFlow version:", tf.__version__)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 动态显存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # optimize_price_lstm(args)
    TCNModel(args).optimize()

    # 5m 15days
    # args["offsetSteps"] = [1, 3, 6]
    # config = {'batch_size': 128, 'seq_len': 32, 'units': 192, 'dropout': 0.4, 'learning_rate': 0.001}

    # 15m 30days
    # args["offsetSteps"] = [1, 2, 4]
    # config = {'batch_size': 128, 'seq_len': 32, 'units': 192, 'dropout': 0.5, 'learning_rate': 0.01}

    # 1h 180days
    # args["offsetSteps"] = [1, 2, 4]
    # config = {'batch_size': 128, 'seq_len': 56, 'units': 160, 'dropout': 0.2, 'learning_rate': 0.01}

    # optimize_trend_lstm(args)

    # lstm = TrendClassifierLstm(args|config)
    # df = lstm.fetch_new_data(
    #     int(datetime.now().timestamp() - 24 * 3600 * args.get("days")) * 1000
    # )

    # print(lstm.init_model(df))
    # print(lstm.predict())


def optimize_tcn(args):
    model = TCNModel(args)
    df = model.fetch_new_data(
        int(datetime.now().timestamp() - 24 * 3600 * args.get("days")) * 1000
    )

    def objective(trial):
        args["filters"] = trial.suggest_int("filters", 16, 64)
        args["kernel_size"] = trial.suggest_int("kernel_size", 2, 4)
        args["dropout_rate"] = trial.suggest_float("dropout_rate", 0.1, 0.4)
        args["n_layers"] = trial.suggest_int("n_layers", 2, 5)

        model = TCNModel(args)
        y_val, y_val_pred = model.init_model(df)
        del model
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        return mean_squared_error(y_val, y_val_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(f"最佳超参数: {best_params}")


def optimize_trend_lstm(args):
    lstm = TrendClassifierLstm(args)
    df = lstm.fetch_new_data(
        int(datetime.now().timestamp() - 24 * 3600 * args.get("days")) * 1000
    )

    def objective(trial):
        # 搜索空间定义
        args["batch_size"] = trial.suggest_categorical("batch_size", [64, 128])
        args["seq_len"] = trial.suggest_categorical(
            "seq_len", [24, 32, 40, 48, 56, 64]
        )  # 4h–10h
        args["units"] = trial.suggest_int(
            "units", 64, 256, step=32
        )  # LSTM 单元数，推荐高一些
        args["dropout"] = trial.suggest_categorical(
            "dropout", [0.1, 0.2, 0.3, 0.4, 0.5]
        )
        args["learning_rate"] = trial.suggest_categorical(
            "learning_rate", [1e-2, 1e-3, 1e-4]
        )

        lstm = TrendClassifierLstm(args)
        val_acc = lstm.init_model(df)
        del lstm
        tf.keras.backend.clear_session()
        return val_acc

    storage_path = (
        f"datas/sqlite/{args['symbol'].replace('/', '-')}_{args['timeframe']}/"
    )
    storage = f"sqlite:///{storage_path}trend.db"
    os.makedirs(os.path.dirname(storage_path), exist_ok=True)
    study = optuna.create_study(direction="maximize", storage=storage)
    study.optimize(objective, n_trials=20)
    # 最优参数
    print("Best Params:", study.best_params)
    # print("Best Epoch:", study.best_trial.user_attrs["best_epoch"])

    # cmd = ["optuna-dashboard", storage, "--port", "9000"]
    # subprocess.Popen(cmd)


# def optimize_price_lstm(args):
#     # 缩放
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(features)

#     def objective(trial):
#         # 搜索空间定义
#         window_size = trial.suggest_categorical(
#             "window_size", [48, 60, 72, 96, 120]
#         )  # 4h–10h
#         args["units"] = trial.suggest_int(
#             "units", 64, 256, step=32
#         )  # LSTM 单元数，推荐高一些
#         args["learning_rate"] = trial.suggest_float(
#             "learning_rate", 1e-4, 5e-3, log=True
#         )
#         args["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
#         args["dropout"] = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)

#         lstm = TrendClassifierLstm(args)

#         model.fit(
#             x_train,
#             y_train,
#             validation_data=(x_val, y_val),
#             epochs=100,
#             batch_size=batch_size,
#             verbose=1,
#             callbacks=[early_stopping],
#         )

#         val_loss = model.evaluate(x_val, y_val, verbose=1)
#         del model

#         tf.keras.backend.clear_session()
#         return val_loss

#     # 启动优化
#     study = optuna.create_study(direction="minimize")
#     study.optimize(objective, n_trials=30)

#     # 最佳参数和损失
#     print("Best params:", study.best_params)
#     print("Best val_loss:", study.best_value)


def show_prediction(args):

    # 5m
    # args["window_size"] = 60
    # args["num_units"] = 32
    # args["learning_rate"] = 0.0026223204209861804
    # args["batch_size"] = 16

    # # 15m
    # args["window_size"] = 36
    # args["num_units"] = 201
    # args["learning_rate"] = 0.0010306858062900388
    # args["batch_size"] = 16

    # [I 2025-05-24 12:27:34,032] Trial 4 finished with value: 7.461294444510713e-05 and parameters: {'window_size': 120, 'units': 192, 'learning_rate': 0.0001606194117961828, 'batch_size': 64, 'dropout': 0.5}. Best is trial 4 with value: 7.461294444510713e-05.
    # BTC/USDT 5m
    args["window_size"] = 72
    args["num_units"] = 192
    args["learning_rate"] = 0.00010981490120101914
    args["batch_size"] = 64
    args["dropout"] = 0.3
    # lstm = BaseLSTMModel(args)
    # lstm.show_prediction_vs_actual(int(args.get("days")), 100)
