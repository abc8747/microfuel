# Quickstart

!!! warning
    The repository is in a pre-alpha state and not ready for production use.

    A convenient PyPI package containing inference-only code with slimmed down dependencies will be released in the future.

## Prerequisites

- Git
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (highly recommended), or `pip`
- Modern GPU for running `triton` kernels (this requirement will be lifted in the future)
- ~10GB disk space for data.

## Installation

Clone the repository and sync the environment. We use `uv` to manage the virtual environment and dependencies.

```sh
git clone https://github.com/abc8747/microfuel
cd microfuel
uv sync --extras cli
```

## Codebase Layout

Understanding the structure will help you navigate the commands:

- `src/microfuel/`: The core library.
    - `datasets/`: Logic for loading raw parquet files (`raw.py`) and generating features (`preprocessed.py`).
    - `model.py`: The GDN, Hypernetwork, and Loss functions.
    - `hacks.py`: JIT-compiled kernel patches for `fla` to support variable-length sequences without recompilation.
- `scripts/`: CLI entry points.
    - `main.py`: The primary interface for all tasks (downloading, processing, training).
    - `plots.py`: Visualization tools for debugging and analysis.

## Step-by-Step Reproduction

### 1. Data Ingestion

The [PRC Data Challenge 2025 data](https://ansperformance.eu/study/data-challenge/dc2025/data.html) is hosted on [S3](https://s3-console.opensky-network.org/). You will need to request a team creation and configure your access credentials in `data/config.toml` (create this file if it doesn't exist, see `data/config.example.toml`).

```sh
# downloads raw parquet files to data/raw/
uv run scripts/main.py download-raw
```

For consistency and to avoid confusion, we rename the data partitions:

```sh
cd data/raw

mv flightlist_train.parquet flight_list_phase1.parquet
mv fuel_train.parquet fuel_phase1.parquet
mv flights_train flights_phase1

mv flightlist_rank.parquet flight_list_phase1_rank.parquet
mv fuel_rank_submission.parquet fuel_phase1_rank.parquet
mv flights_rank flights_phase1_rank

mv flightlist_final.parquet flight_list_phase2_rank.parquet
mv fuel_train.parquet fuel_phase2_rank.parquet
mv flights_final flights_phase2_rank
```

### 2. Preprocessing

This step applies the Kalman Filter/RTS Smoother to the raw ADS-B points to generate clean inputs. It also generates the train/validation splits based on stratified sampling of aircraft types and flight durations.

```sh
# smoothed trajectory vectors (heavy CPU usage, ~30 minutes)
uv run scripts/main.py create-dataset --partition phase1

# generate normalisation statistics (mean/std, ~5 minutes)
uv run scripts/main.py create-stats --partition phase1

# create stratified splits
uv run scripts/main.py create-splits --partition phase1
```

### 3. Training

Launch the training loop. The model uses [`wandb`](https://github.com/wandb/wandb) for logging.

```sh
uv run scripts/main.py train \
    --partition phase1 \
    --exp-name "quickstart-gdn-v1" \
    --batch-size 64 \
    --lr 4e-4 \
    --epochs 20 \
    --beta 0.999 \
    --loss-type rmse_kg
```

- `--beta`: The [Class-Balanced Loss hyperparameter](./problem.md#class-imbalance). Higher values (e.g., 0.999) heavily upweight rare aircraft types.
- `--loss-type`: To avoid long tail distributions, the model outputs the *average fuel burn rate* over the segment instead of the total fuel burnt in that segment. `rmse_kg` effectively optimises against both for training stability.

### 4. Evaluation

To evaluate a specific checkpoint on the validation set:

```sh
uv run scripts/main.py evaluate \
    data/checkpoints/quickstart-gdn-v1/step00500_123.45.pt \
    --partition phase1 \
    --split validation
```

This will generate a parquet file in `data/predictions/` containing ground truth vs. predicted values for analysis.
