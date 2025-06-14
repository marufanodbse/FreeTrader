# predictor/service.py

import tensorflow as tf
import joblib
import numpy as np
import json
from fastapi import FastAPI, Query
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

from common.util import get_current_kline_timestamp
from models.lstm.trend_classifier_model import TrendClassifierLstm
from cachetools import TTLCache, cached

models = {}


def start_trains(args):
    with open(args["config"], "r") as f:
        config = json.load(f)
    
    scheduler = BackgroundScheduler()
    pairConfigs = config["trains"]
    for eachArgs in pairConfigs:
        model = TrendClassifierLstm(
            {
                "exchange": "binance",
                "proxy": "http://127.0.0.1:1087",
            }
            | eachArgs
        )
        key = f'{eachArgs["symbol"]}_{eachArgs["timeframe"]}'
        models[key] = model
        unit = eachArgs["timeframe"][-1]
        value = eachArgs["timeframe"][:-1]
        model.init_model()

        # if unit == "m":
        #     cron_args = {"minute": f"0/{value}", "minute": "0/{value}", "second": 5}
        # elif unit == "h":
        #     cron_args = {"hour": f"0/{value}", "minute": 0, "second": 10}
        # else:
        #     raise ValueError(f"Unsupported timeframe: {eachArgs["timeframe"]}")
        # scheduler.add_job(model.predict, "cron", **cron_args)

        # if timeframe.endswith('m'):
        #     scheduler.add_job(model.run_train, "cron", minute=f"0/{timeframe.replace('m', '')}", second=10)
        # elif timeframe.endswith('h'):
        #     scheduler.add_job(model.run_train, "cron", hour="0/{timeframe.replace('h', '')}", second=10)
    # scheduler.start()

    app = FastAPI()

    @app.get("/predict")
    def predict_price(
        symbol: str = Query(...),
        timeframe: str = Query(...),
    ):
        return predict(symbol, timeframe)

    cache = TTLCache(maxsize=100, ttl=60)

    def make_key(symbol, timeframe):
        timestamp = get_current_kline_timestamp(timeframe)
        return f"{symbol}_{timeframe}_{timestamp}"

    @cached(
        cache,
        key=lambda symbol, timeframe: make_key(symbol, timeframe),
    )
    def predict(symbol, timeframe):
        model: TrendClassifierLstm = models.get(f"{symbol}_{timeframe}")
        if model is None:
            raise RuntimeError(f"nofund model[{symbol}_{timeframe}]")
        reuslt = model.predict()
        print(reuslt)
        return reuslt

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.get("port", 8000))
