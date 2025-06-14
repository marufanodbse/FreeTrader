from datetime import datetime
import pandas as pd
import numpy as np
import talib
import xgboost as xgb
from sklearn.model_selection import train_test_split # 再次提醒：这里仅为演示，实际应按时间分割
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from common.util import parse_interval_to_microseconds
from finance.exchange import Exchange
# from ta import add_all_ta_features # 用于技术指标计算的库，强烈推荐

def init_model():
    # 初始化 XGBoost 分类器
    # objective='multi:softmax' 用于多分类任务，如果只有2分类用 'binary:logistic'
    # num_class 设置为你的类别数量
    # eval_metric='mlogloss' 是多分类的评估指标，accuracy 也可以
    # tree_method='hist' 可以加速训练
    model = xgb.XGBClassifier(objective='multi:softprob',
                            num_class=3, # 上涨，下跌，横盘
                            eval_metric='mlogloss',
                            use_label_encoder=False, # 避免未来版本警告
                            n_estimators=1000, # 树的数量，可以设置大一点，配合 early stopping
                            learning_rate=0.05, # 学习率
                            max_depth=5, # 树的最大深度
                            subsample=0.8, # 每次训练样本的比例
                            colsample_bytree=0.8, # 每次训练特征的比例
                            gamma=0.1, # 降低模型复杂度的参数
                            random_state=42)
    

class TrendClassifierLstm:
    def __init__(self, args):
        config = {}
        if "proxy" in args:
            config["proxies"] = {
                "http": args["proxy"],
                "https": args["proxy"],
            }
        self.exchange = Exchange(args.get("exchange", "binance"), config)
        self.symbol = args.get("symbol")
        self.timeframe = args.get("timeframe")

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
    
    def label_trend(df, future_offset=3, up_th=0.003, down_th=-0.003):
        future_close = df["close"].shift(-future_offset)
        pct_change = (future_close - df["close"]) / df["close"]

        df["trend"] = np.where(
            pct_change > up_th, 1, np.where(pct_change < down_th, -1, 0)
        )
        return df
    
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

        df = self.label_trend(df)
        df = df.dropna()
        df = df[["close", "rsi", "macd", "macd_signal", "macd_hist", "atr"]]
        return self.label_trend(df)
    
    # 训练模型，使用 early stopping
    # early_stopping_rounds 意味着如果在指定轮数内验证集的性能没有提升，训练就停止
    def train(self):
        since = int(datetime.now().timestamp() - 24 * 3600 * 90) * 1000
        df_data = self.fetch_new_data(since)


        self.build_features()
        # features = df_data.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume', 'target'])
        # target = df_data['target']

        # # --- 数据标准化 (可选，但通常推荐) ---
        # scaler = StandardScaler()
        # scaled_features = scaler.fit_transform(features)
        # X = pd.DataFrame(scaled_features, columns=features.columns)
        # y = target

        # # --- 划分训练集、验证集、测试集 (按时间顺序) ---
        # # train_test_split 默认是随机的，实际应用请使用时间分割
        # # 这里为了示例方便，先用train_test_split，但请务必替换
        # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
        # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False) # 验证集和测试集各占总数据的15%

        # print(f"\n训练集: X_train={X_train.shape}, y_train={y_train.shape}")
        # print(f"验证集: X_val={X_val.shape}, y_val={y_val.shape}")
        # print(f"测试集: X_test={X_test.shape}, y_test={y_test.shape}")

        # model.fit(X_train, y_train,
        #     eval_set=[(X_val, y_val)],
        #     early_stopping_rounds=50,
        #     verbose=False) # 设置为True可以看到训练过程中的评估指标
