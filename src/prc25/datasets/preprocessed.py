from __future__ import annotations

import json
import logging
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, get_args

import numpy as np
import polars as pl
from rich.progress import track

from .. import PATH_PREPROCESSED, Partition
from . import raw

if TYPE_CHECKING:
    from datetime import datetime
    from multiprocessing.synchronize import Event
    from typing import TypeAlias

logger = logging.getLogger(__name__)
AcType: TypeAlias = str
Feature = Literal["time_since_takeoff", "log_dt_1", "altitude", "groundspeed", "vertical_rate"]
FEATURES = get_args(Feature)


def _np_interpolate(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    valid_mask = ~np.isnan(y)
    if not np.any(valid_mask):
        return y
    return np.interp(x, x[valid_mask], y[valid_mask])


def make_trajectories(partition: Partition):
    PATH_PREPROCESSED.mkdir(exist_ok=True, parents=True)

    flight_list_lf = raw.scan_flight_list(partition)
    fuel_lf = raw.scan_fuel(partition)

    segments_df = flight_list_lf.join(fuel_lf, on="flight_id").sort("flight_id").collect()
    logger.info(f"found {len(segments_df)} segments in partition `{partition}`")

    processed_trajectories: list[pl.LazyFrame] = []

    flight_groups = list(segments_df.group_by("flight_id"))
    for (flight_id,), segments in track(
        flight_groups,
        description="processing flights",
        total=len(flight_groups),
    ):
        traj_lf = raw.scan_trajectory(flight_id, partition)
        assert traj_lf is not None

        full_traj_df = traj_lf.sort("timestamp").collect()
        assert full_traj_df.height > 2

        timestamp_s = full_traj_df["timestamp"].dt.epoch(time_unit="s").to_numpy()
        alt_raw = full_traj_df["altitude"].to_numpy()
        gs_raw = full_traj_df["groundspeed"].to_numpy()
        vs_raw = full_traj_df["vertical_rate"].to_numpy()

        alt_outlier_mask = (alt_raw > 50000) | np.isnan(alt_raw)
        gs_outlier_mask = (gs_raw > 800) | np.isnan(gs_raw)
        vs_outlier_mask = (np.abs(vs_raw) > 8000) | np.isnan(vs_raw)

        alt_with_nan = np.where(alt_outlier_mask, np.nan, alt_raw)
        gs_with_nan = np.where(gs_outlier_mask, np.nan, gs_raw)
        vs_with_nan = np.where(vs_outlier_mask, np.nan, vs_raw)

        alt_interp = _np_interpolate(alt_with_nan, timestamp_s)
        gs_interp = _np_interpolate(gs_with_nan, timestamp_s)
        vs_interp = _np_interpolate(vs_with_nan, timestamp_s)

        processed_traj_df = pl.DataFrame(
            {
                "timestamp": full_traj_df["timestamp"],
                "altitude": alt_interp,
                "altitude_is_outlier": alt_outlier_mask,
                "groundspeed": gs_interp,
                "groundspeed_is_outlier": gs_outlier_mask,
                "vertical_rate": vs_interp,
                "vertical_rate_is_outlier": vs_outlier_mask,
            }
        )

        segment_expr = pl.lit(None, dtype=pl.UInt32)
        for row in segments.iter_rows(named=True):
            segment_expr = (
                pl.when(pl.col("timestamp").is_between(row["start"], row["end"]))
                .then(pl.lit(row["idx"], dtype=pl.UInt32))
                .otherwise(segment_expr)
            )

        takeoff_time: datetime = segments.select(pl.col("takeoff").first()).item()

        final_df = (
            processed_traj_df.lazy()
            .with_columns(
                pl.lit(flight_id).alias("flight_id"),
                segment_expr.alias("segment_id"),
                (
                    (pl.col("timestamp") - takeoff_time)
                    .dt.total_seconds(fractional=True)
                    .alias("time_since_takeoff")
                ),
                (
                    (
                        pl.col("timestamp")
                        .diff()
                        .dt.total_seconds(fractional=True)
                        .fill_null(0.0)  # first data point has no dt
                    )
                    + 1.0
                )
                .log()
                .alias("log_dt_1"),
            )
            .select(
                "flight_id",
                "segment_id",
                "time_since_takeoff",
                "log_dt_1",
                "altitude",
                "altitude_is_outlier",
                "groundspeed",
                "groundspeed_is_outlier",
                "vertical_rate",
                "vertical_rate_is_outlier",
            )
        )
        processed_trajectories.append(final_df)

    output_path = PATH_PREPROCESSED / f"trajectories_{partition}.parquet"

    stop_event = multiprocessing.Event()
    monitor_process = multiprocessing.Process(
        target=_monitor_file_size, args=(output_path, stop_event)
    )
    monitor_process.start()

    try:
        pl.concat(processed_trajectories).sink_parquet(output_path)
    finally:
        stop_event.set()
        monitor_process.join()

    logger.info(f"wrote combined state vectors to {output_path}")


def _monitor_file_size(path: Path, stop_event: Event, *, dt: float = 0.2, name: str = "writing"):
    import time

    from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
    ) as progress:
        task = progress.add_task(name, total=None)
        while not stop_event.is_set():
            if path.exists():
                size = path.stat().st_size
                progress.update(task, completed=size)
            time.sleep(dt)


class Stat(TypedDict):
    mean: float
    std: float


class SplitData(TypedDict):
    train_flight_ids: list[str]
    validation_flight_ids: list[str]
    standardisation_stats: dict[Feature, Stat]


def make_train_validation_split(
    partition: Partition,
    train_split: float = 0.9,
    seed: int = 13,
):
    flight_list_lf = raw.scan_flight_list(partition)
    fuel_lf = raw.scan_fuel(partition)

    flight_ids = (
        flight_list_lf.join(fuel_lf, on="flight_id")
        .select("flight_id")
        .unique()
        .collect()["flight_id"]
        .shuffle(seed=seed)
        .to_list()
    )

    split_idx = int(len(flight_ids) * train_split)
    flight_ids_train = flight_ids[:split_idx]
    flight_ids_validation = flight_ids[split_idx:]

    traj_lf = pl.scan_parquet(PATH_PREPROCESSED / f"trajectories_{partition}.parquet")

    train_segments_lf = traj_lf.filter(
        pl.col("flight_id").is_in(flight_ids_train) & pl.col("segment_id").is_not_null()
    )

    stats_df = train_segments_lf.select(
        [pl.mean(col).alias(f"{col}_mean") for col in FEATURES]
        + [pl.std(col).alias(f"{col}_std") for col in FEATURES]
    ).collect()

    standardisation_stats: dict[Feature, Stat] = {}
    row = stats_df.row(0, named=True)
    for feature in FEATURES:
        standardisation_stats[feature] = {
            "mean": row[f"{feature}_mean"],
            "std": row[f"{feature}_std"],
        }

    split_data: SplitData = {
        "train_flight_ids": flight_ids_train,
        "validation_flight_ids": flight_ids_validation,
        "standardisation_stats": standardisation_stats,
    }

    output_path = PATH_PREPROCESSED / f"split_{partition}.json"
    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=2)
    logger.info(f"wrote split and stats to {output_path}")


#
# wrappers
#


def train_validation_split(
    partition: Partition,
) -> SplitData:
    with open(PATH_PREPROCESSED / f"split_{partition}.json") as f:
        return json.load(f)  # type: ignore
