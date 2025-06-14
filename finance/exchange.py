from datetime import datetime
import sys
import time
import ccxt
from ccxt.base import types
from typing import List
from apscheduler.schedulers.background import BackgroundScheduler
from concurrent.futures import ThreadPoolExecutor

from common.util import format_timestamp, parse_interval_to_microseconds, run_tasks


class Exchange:

    def __init__(self, name, config):
        config["enableRateLimit"] = True
        if name not in ccxt.exchanges:
            raise ValueError(f"不支持的交易所: {name}")
        exchange_class = getattr(ccxt, name)
        self.exchange = exchange_class(config)
        pass

    def fetch_ohlcv(self, symbol, timeframe, since=None, count=1000) -> List[list]:
        return self.exchange.fetch_ohlcv(symbol, timeframe, since, count)

    def fetch_ticker(self, symbol) -> types.Ticker:
        return self.exchange.fetch_ticker(symbol)

    def fetch_balance(self, symbol) -> types.Balance:
        return self.exchange.fetch_balance(symbol)

    def fetch_order(self, id, symbol=None) -> types.Order:
        return self.exchange.fetch_order(id, symbol)

    def fetch_orders(self, symbol, status=None, since=0, limit=10) -> List[types.Order]:
        if status is None:
            return self.exchange.fetch_orders(symbol, since, limit)
        elif status == "open":
            return self.exchange.fetch_open_orders(symbol, since, limit)
        elif status == "closed":
            return self.exchange.fetch_closed_orders(symbol, since, limit)
        elif status == "canceled":
            return self.exchange.fetch_canceled_orders(symbol, since, limit)

    def create_order(self, symbol, side, amount, price=None, positionSide=None):
        type = "limit"
        if price is None:
            type = "market"

        defaultType = self.exchange.options["defaultType"]
        params = {}

        if defaultType == "future":
            params["type"] = defaultType
            if positionSide is None:
                params["positionSide"] = "LONG" if side == "buy" else "SHORT"
            else:
                params["positionSide"] = positionSide

        return self.exchange.create_order(symbol, type, side, amount, price, params)

    def cancel_order(self, id):
        return self.exchange.cancel_order(id)

    def cancel_all_orders(self, symbol):
        return self.exchange.cancel_all_orders(symbol)

    def fetch_position(self, symbol):
        return self.exchange.fetch_position(symbol)

    def set_position_mode(self, hedged):
        return self.exchange.set_position_mode(hedged)

    def fetch_trading_fee(self, symbol):
        return self.exchange.fetch_trading_fee(symbol)

    def fetch_with_retry(self, symbol, timeframe, since, count=1000):
        try:
            
            ohlcvs = self.fetch_ohlcv(symbol, timeframe, since, count)
            # print(f"fatch since: {format_timestamp(since)}, len: {len(ohlcvs)}")
            return ohlcvs
        except Exception as e:
            print(f"Failed to fetch[{since}]: {e}")
            time.sleep(0.01)
            return self.fetch_with_retry(symbol, timeframe, since)

    def fetch_ohlcv_batch(self, symbol, timeframe, since, count):
        args = []
        while count > 0:
            args.append(
                (
                    symbol,
                    timeframe,
                    since,
                    1000,
                )
            )
            since += 1000 * parse_interval_to_microseconds(timeframe)
            count -= 1000
            if since > datetime.now().timestamp() * 1000:
                break
        results = run_tasks(self.fetch_with_retry, args)
        result = [item for group in results for item in group]
        return result
