import logging
from typing import Literal

import typer
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command()
def download_raw():
    from prc25.datasets.raw import download_from_s3, load_config

    config = load_config()
    download_from_s3(config["bucket_access_key"], config["bucket_access_secret"])


@app.command()
def create_dataset(
    partition: Literal["train", "rank"],
):
    from prc25.datasets.preprocessed import create_preprocessed

    create_preprocessed(partition=partition)


@app.command()
def train():
    pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=False)]
    )
    app()
