from argparse import _ArgumentGroup, ArgumentParser, Namespace
from typing import Any

from common import constants
from core.kline_downloader import download_data
from core.service import start_trains
from optimize.optimizer import optimize, show_prediction


class Arg:
    # Optional CLI arguments
    def __init__(self, *args, **kwargs):
        self.cli = args
        self.kwargs = kwargs


# List of available command line options
AVAILABLE_CLI_OPTIONS = {
    "version": Arg(
        "-V",
        "--version",
        help="show program's version number and exit",
        action="store_true",
    ),
    "config": Arg(
        "-c",
        "--config",
        help=f"Specify configuration file (default: `userdir/{constants.DEFAULT_CONFIG}` "
        f"or `config.yaml` whichever exists). "
        f"Multiple --config options may be used. "
        f"Can be set to `-` to read config from stdin.",
        # action="append",
        metavar="PATH",
    ),
    "proxy": Arg(
        "--proxy",
        help="set proxy for ccxt client",
    ),
    "symbol": Arg(
        "--symbol",
        help="symbol",
    ),
    "timeframe": Arg(
        "-i",
        "--timeframe",
        help="Specify timeframe (`1m`, `5m`, `30m`, `1h`, `1d`).",
    ),
    "since": Arg(
        "--since",
        help="from date",
    ),
    "exchange": Arg(
        "--exchange",
        help="exchange name",
    ),
    "windowsize": Arg(
        "--windowsize",
        type=int,
        help="exchange name",
    ),
    "units": Arg(
        "--units",
        type=int,
        help="exchange name",
    ),
    "learningrate": Arg(
        "--learningrate",
        help="exchange name",
    ),
    "batchsize": Arg(
        "--batchsize",
        type=int,
        help="exchange name",
    ),
    "days": Arg(
        "--days",
        type=int,
        help="days",
    ),
}

ARGS_PAIR = ["symbol", "timeframe"]
ARGS_EXCHANGE = ["exchange", "proxy"]
ARGS_MODEL = ["config"]
ARGS_DOWNLOAD_DATA = ["since"] + ARGS_EXCHANGE + ARGS_PAIR
ARGS_OPTIMIZE = ["days"] + ARGS_EXCHANGE + ARGS_PAIR
ARGS_SHOW_PREDICTION = (
    [
        "windowsize",
        "units",
        "learningrate",
        "batchsize",
        "days"
    ]
    + ARGS_EXCHANGE
    + ARGS_PAIR
)


class Arguments:
    """
    Arguments Class. Manage the arguments received by the cli
    """

    def __init__(self, args: list[str] | None) -> None:
        self.args = args
        self._parsed_arg: Namespace | None = None

    def get_parsed_arg(self) -> dict[str, Any]:
        """
        Return the list of arguments
        :return: List[str] List of arguments
        """
        if self._parsed_arg is None:
            self._build_subcommands()
            self._parsed_arg = self._parse_args()

        return vars(self._parsed_arg)

    def _parse_args(self) -> Namespace:
        """
        Parses given arguments and returns an argparse Namespace instance.
        """
        parsed_arg = self.parser.parse_args(self.args)
        return parsed_arg

    def _build_args(
        self, optionlist: list[str], parser: ArgumentParser | _ArgumentGroup
    ) -> None:
        for val in optionlist:
            opt = AVAILABLE_CLI_OPTIONS[val]
            parser.add_argument(*opt.cli, dest=val, **opt.kwargs)

    def _build_subcommands(self) -> None:
        """
        Builds and attaches all subcommands.
        :return: None
        """
        self.parser = ArgumentParser(
            prog="freqtrade", description="Free, open source crypto trading bot"
        )
        self._build_args(optionlist=["version"], parser=self.parser)

        subparsers = self.parser.add_subparsers(
            dest="command",
            # Use custom message when no subhandler is added
            # shown from `main.py`
            # required=True
        )

        # Add

        # Add trade subcommand
        model_cmd = subparsers.add_parser("train", help="init and train model.", parents=[])
        model_cmd.set_defaults(func=start_trains)
        self._build_args(optionlist=ARGS_MODEL, parser=model_cmd)

        # Add download-data subcommand
        download_data_cmd = subparsers.add_parser(
            "download-data",
            help="Download ohlcv data.",
        )
        download_data_cmd.set_defaults(func=download_data)
        self._build_args(optionlist=ARGS_DOWNLOAD_DATA, parser=download_data_cmd)

        # Add optimize subcommand
        optimize_cmd = subparsers.add_parser(
            "optimize",
            help="optimize LSTM params.",
        )
        optimize_cmd.set_defaults(func=optimize)
        self._build_args(optionlist=ARGS_OPTIMIZE, parser=optimize_cmd)

        # Add showPrediction subcommand
        show_prediction_cmd = subparsers.add_parser(
            "showPrediction",
            help="show prediction vs actual",
        )
        show_prediction_cmd.set_defaults(func=show_prediction)
        self._build_args(optionlist=ARGS_SHOW_PREDICTION, parser=show_prediction_cmd)
