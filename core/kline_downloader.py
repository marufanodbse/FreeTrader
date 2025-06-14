import os
import time
import ccxt
import numpy as np
import pandas as pd
from datetime import datetime

from common.util import (
    format_timestamp,
    parse_interval_to_microseconds,
)
from finance.exchange import Exchange

exchange = Exchange(
    "binance",
    {
        "proxies": {
            "http": "http://127.0.0.1:1087",
            "https": "http://127.0.0.1:1087",
        }
    },
)


def download_data(args):
    # ohlcvs = exchange.fetch_ohlcv_batch(args["symbol"], '1m', 1581213600000, 10000)
    # df = pd.DataFrame(
    #     ohlcvs, columns=["timestamp", "open", "high", "low", "close", "volume"]
    # )
    # print(df[0:])
    # check_timestamp_continuity(df)
    download_data_loop(args["symbol"], args["timeframe"], int(args["since"]))

def download_data_loop(symbol, timeframe, since=None):
    filename = f"datas/finances/{symbol.replace('/','-')}_{timeframe}.csv"
    if os.path.exists(filename):
        df_old = pd.read_csv(filename)
        if not df_old.empty:
            since = df_old["timestamp"].max() + parse_interval_to_microseconds(
                timeframe
            )
    else:
        df_old = pd.DataFrame()

    ohlcvs = exchange.fetch_ohlcv_batch(symbol, timeframe, since, 10000)
    df_new = pd.DataFrame(
        ohlcvs, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    # if ohlcvs[0][0] != since or check_timestamp_continuity(df_new) == False:
    #     print(f"{since} {ohlcvs[0][0]} datas error")
    #     return
    
    if not ohlcvs or len(ohlcvs) == 0:
        # check_timestamp_continuity(df_old)
        print(f"[{datetime.now()}] No new data for {symbol} {timeframe}")
        return
    print(
        f"Fetching[{symbol}_{timeframe} {len(ohlcvs)}] from: {format_timestamp(ohlcvs[0][0])} to: {format_timestamp(ohlcvs[-1][0])}"
    )

    
    # df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")

    df_all = pd.concat([df_old, df_new]) if not df_old.empty else df_new
    df_all = df_all.drop_duplicates(subset="timestamp").sort_values("timestamp")
    # if check_timestamp_continuity(df_all) == False:
    #     return
    df_all.to_csv(filename, index=False)
    print(f"Fetching[{symbol}_{timeframe}] Updated total rows: {len(df_all)}")

    if len(df_new) >= 10000:
        download_data_loop(symbol, timeframe)


# def check_data(args):
#     filename = f"datas/finances/{args["symbol"].replace('/','-')}_{args["timeframe"]}.csv"
#     df = pd.read_csv(filename)
#     check_timestamp_continuity(df)

def check_timestamp_continuity(df, column="timestamp", interval=300000.0):
   
    ts = df[column].reset_index(drop=True)
    diffs = ts.diff().dropna()

    if (abs(diffs - interval) > 1).any():
        discontinuous_idx = diffs[diffs != interval].index
        print("不连续的索引位置：", discontinuous_idx.tolist())
        return False
    else:
        return True
    # else:
        # print(f"列 {column} 中的 timestamp 是连续的")

# def download_data_loop(symbol, timeframe, since=None):
#     filename = f"datas/finances/{symbol.replace('/','-')}_{timeframe}.csv"
#     if os.path.exists(filename):
#         df_old = pd.read_csv(filename)
#         if not df_old.empty:
#             since = df_old["timestamp"].max() + parse_interval_to_microseconds(
#                 timeframe
#             )
#     else:
#         df_old = pd.DataFrame()
#     print(f"since:{since}")
#     all_data = []
#     while True:
#         try:
#             ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
#             if not ohlcv:
#                 break
#             all_data += ohlcv
#             since = ohlcv[-1][0] + parse_interval_to_microseconds(timeframe)
#             print(
#                 f"Fetching[{symbol}_{timeframe}] from: {format_timestamp(ohlcv[0][0])} to: {format_timestamp(ohlcv[-1][0])}"
#             )
#             if len(ohlcv) < 1000:
#                 break

#             if len(all_data) >= 50000:
#                 break
#             # time.sleep(0.5)
#         except Exception as e:
#             print(f"[{datetime.now()}] Error fetching data: {e}")
#             break

#     if not all_data:
#         print(f"[{datetime.now()}] No new data for {symbol} {timeframe}")
#         return

#     df_new = pd.DataFrame(
#         all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
#     )
#     # df_new["timestamp"] = pd.to_datetime(df_new["timestamp"], unit="ms")

#     df_all = pd.concat([df_old, df_new]) if not df_old.empty else df_new
#     df_all = df_all.drop_duplicates(subset="timestamp").sort_values("timestamp")
#     df_all.to_csv(filename, index=False)
#     print(f"Fetching[{symbol}_{timeframe}] Updated total rows: {len(df_all)}")

#     download_data_loop(symbol, timeframe)
