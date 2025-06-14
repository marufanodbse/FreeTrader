import logging
import sys
from logging.handlers import TimedRotatingFileHandler
import os

def setup_logger(
    name: str,
    log_file: str = "app.log",
    level: int = logging.INFO,
    when: str = "midnight",   # 每天 0 点创建新日志文件
    interval: int = 1,        # 间隔 1 个单位（天）
    backup_count: int = 7     # 最多保留 7 天的日志文件
) -> logging.Logger:
    """
    设置一个每天生成一个新日志文件，同时输出到控制台的 logger。
    
    :param name: logger 名称
    :param log_file: 日志文件前缀
    :param level: 日志等级
    :param when: 时间单位（默认 'midnight' 表示每天切）
    :param interval: 多久轮转一次（默认 1）
    :param backup_count: 最多保留几个旧日志文件
    :return: 配置好的 logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger  # 避免重复添加 handler

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 文件日志：每天生成一个新文件
    file_handler = TimedRotatingFileHandler(
        log_file,
        when=when,
        interval=interval,
        backupCount=backup_count,
        encoding='utf-8',
        utc=False  # True 表示使用 UTC 时间，False 表示本地时间
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台日志
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
