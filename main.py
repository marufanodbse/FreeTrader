# main.py
import os
import logging
import sys
import ccxt
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime

from typing import Any
from sklearn.preprocessing import MinMaxScaler
from commands.arguments import Arguments
from common.constants import VERSION
from common.exceptions import ConfigurationError, FreetraderException
from finance.exchange import Exchange

from system.gc_setup import gc_set_threshold
from system.version_info import print_version_info

logger = logging.getLogger("freqtrade")

def main(sysargv: list[str] | None = None) -> None:
    """
    This function will initiate the bot and start the trading loop.
    :return: None
    """
    return_code: Any = 1
    try:
        arguments = Arguments(sysargv)
        args = arguments.get_parsed_arg()

        if args.get("version") or args.get("version_main"):
            print_version_info()
            return_code = 0
        elif "func" in args:
            logger.info(f"freetrader {VERSION}")
            gc_set_threshold()
            return_code = args["func"](args)
    except SystemExit as e:  # pragma: no cover
        return_code = e
    except KeyboardInterrupt:
        logger.info("SIGINT received, aborting ...")
        return_code = 0
    except ConfigurationError as e:
        logger.error(
            f"Configuration error: {e}\n"
            f"Please make sure to review the documentation"
        )
    except FreetraderException as e:
        logger.error(str(e))
        return_code = 2
    except Exception:
        logger.exception("Fatal exception!")
    finally:
        sys.exit(return_code)


# Step 3: Create dataset for LSTM



if __name__ == "__main__":
    main()
    
    