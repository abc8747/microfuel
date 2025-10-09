# PRC-DataChallenge

An open machine learning model to infer the actual fuel burn of a flown flight. For more details, please check the [website](https://ansperformance.eu/study/data-challenge/)

## Installation

Clone the repository, install and activate your virtual environment.

=== "pip"

     ```sh
     pip3 install virtualenv --break-system-packages
     virtualenv .venv
     . .venv/bin/activate
     pip3 install ".[dev]"
     ```

=== "uv"

     ```sh
     pip3 install uv --break-system-packages
     uv venv
     uv sync --all-extras
     ```

     Activate your virtual environment with `. .venv/bin/activate`, alternatively, use `uv run python3 scripts/{FILENAME}.py` to run a particular script.

This repo does not come with raw data. See [here](./data.md) for instructions to set it up.

## Quickstart

Code is structured as follows:

- `src/prc25`: core library code
- `scripts`: CLI-based tools

## Documentation

In the root directory, run

    ```sh
    mkdocs build
    ```

and navigate to <https://localhost:8080/prc-datachallenge/>.

## Python Code style

1. Follow [Black](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html) style. 
2. Use typed code whenever possible.
3. Prefer pure functions over OOP.
4. Prefer scripts over notebooks. If you want interactive cells, use [`#%%` in `scripts`](https://code.visualstudio.com/docs/python/jupyter-support-py).

We use the following tools to check the style on each push:

- [Ruff](https://github.com/astral-sh/ruff) for linting,
- [MyPy](https://github.com/python/mypy) for type checking

Locally, run the following before committing:

    ```sh
    just fmt
    ```

Recommended VSCode extensions: `charliermarsh.ruff`, `matangover.mypy`, `usernamehw.errorlens`, `ms-toolsai.jupyter`

### Leaderboard

To make submissions, use `data/credentials.example.toml` as the template.