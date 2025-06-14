from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


def run_tasks(task, args_list):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda args: task(*args), args_list))
    return results


def get_current_kline_timestamp(timeframe):
    timestamp = int(datetime.now().timestamp() * 1000)
    return timestamp - timestamp % parse_interval_to_seconds(timeframe)


def parse_interval_to_seconds(interval_str):
    unit = interval_str[-1]
    value = int(interval_str[:-1])

    unit_map = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800,
    }

    if unit not in unit_map:
        raise ValueError(f"Unsupported time unit: {unit}")

    return value * unit_map[unit]


def parse_interval_to_microseconds(interval_str):
    return parse_interval_to_seconds(interval_str) * 1000


def format_timestamp(ts, fmt="%Y-%m-%d %H:%M:%S"):
    """
    将时间戳（秒或毫秒）转换为格式化时间字符串。

    参数:
        ts: int or float,秒级或毫秒级时间戳
        fmt: str,格式化字符串(默认为 "%Y-%m-%d %H:%M:%S")

    返回:
        str,格式化后的时间字符串
    """
    if ts > 1e12:  # 毫秒级
        ts = ts / 1000
    dt = datetime.fromtimestamp(ts)
    return dt.strftime(fmt)


def is_hour_start(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000)
    return dt.minute == 0 and dt.second == 0 and dt.microsecond == 0


def is_day_start(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000)
    return dt.hour == 0 and dt.minute == 0 and dt.second == 0 and dt.microsecond == 0


def is_week_start(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000)
    return is_day_start(ts_ms) and dt.weekday() == 0  # Monday is 0


def is_month_start(ts_ms):
    dt = datetime.fromtimestamp(ts_ms / 1000)
    return is_day_start(ts_ms) and dt.day == 1
