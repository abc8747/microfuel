import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def download_raw():
    from prc25.datasets.raw import download_from_s3, load_config

    config = load_config()
    download_from_s3(config["bucket_access_key"], config["bucket_access_secret"])


@app.command()
def check():
    pass


if __name__ == "__main__":
    app()
