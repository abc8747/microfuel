# PRC-DataChallenge

Open ML model to for learning the underlying continuous-time dynamics of an aircraft from a sparse, irregularly-sampled multivariate time series to accurately predict an integrated quantity (total fuel burn) over arbitrary future time intervals.

[[Challenge Link](https://ansperformance.eu/study/data-challenge/)]

## Installation

Clone the repository, install and activate your virtual environment.

With `pip`:

```sh
pip3 install virtualenv --break-system-packages
virtualenv .venv
. .venv/bin/activate
pip3 install ".[dev]"
```

Or `uv`:

```sh
pip3 install uv --break-system-packages
uv venv
uv sync --all-extras
```

Activate your virtual environment with `. .venv/bin/activate`, alternatively, use `uv run python3 scripts/{FILENAME}.py` to run a particular script.

This repo does not come with raw data.
You may want to install the MinIO client (`mc`), necessary for pulling and pushing to OpenSky's S3 bucket.

## Quickstart

Code is structured as follows:

- `src/prc25`: core library code
- `scripts`: CLI-based tools

## Documentation

In the root directory, run

```sh
mkdocs build
```

and navigate to <https://localhost:8080/prc25/>.

## Python Code style

1. Prefer pure functions over OOP.
2. Prefer scripts over notebooks. If you want interactive cells, use [`#%%` in `scripts`](https://code.visualstudio.com/docs/python/jupyter-support-py).

We use the following tools to check the style on each push:

- [Ruff](https://github.com/astral-sh/ruff) for linting,
- [MyPy](https://github.com/python/mypy) for type checking

Locally, run the following before committing:

```sh
just fmt
```
