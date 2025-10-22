from __future__ import annotations

import json
import logging
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict, get_args

import numpy as np
import polars as pl
from rich.progress import track

from .. import PATH_PREPROCESSED, Partition, Split
from . import raw

if TYPE_CHECKING:
    import datetime
    from multiprocessing.synchronize import Event
    from typing import Collection, Iterator, TypeAlias

logger = logging.getLogger(__name__)
AcType: TypeAlias = str
FlightId: TypeAlias = str
TrajectoryFeature = Literal["time_since_takeoff", "altitude", "groundspeed", "vertical_rate"]
TRAJECTORY_FEATURES = get_args(TrajectoryFeature)
# SegmentFeature = Literal["time_since_start", "time_since_end"]
# SEGMENT_FEATURES = get_args(SegmentFeature)


def _np_interpolate(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    valid_mask = ~np.isnan(y)
    if not np.any(valid_mask):
        return y
    return np.interp(x, x[valid_mask], y[valid_mask])


def make_trajectories(partition: Partition, train_split: float = 0.8, seed: int = 13):
    """Creates train/validation split of preprocessed trajectories.

    Everything related to segments (e.g. whether a particular state vector is within [start, end])
    should be handled elsewhere. This function processes the *entire* trajectory.
    """
    PATH_PREPROCESSED.mkdir(exist_ok=True, parents=True)

    flight_list_lf = raw.scan_flight_list(partition)
    fuel_lf = raw.scan_fuel(partition)

    flight_ids_with_fuel = (
        flight_list_lf.join(fuel_lf, on="flight_id")
        .select("flight_id")
        .unique()
        .sort("flight_id")
        .collect()["flight_id"]
        .shuffle(seed=seed)
        .to_list()
    )
    logger.info(
        f"found {len(flight_ids_with_fuel)} flights with fuel data in partition `{partition}`"
    )

    split_idx = int(len(flight_ids_with_fuel) * train_split)
    flight_ids_train = flight_ids_with_fuel[:split_idx]
    flight_ids_validation = flight_ids_with_fuel[split_idx:]
    train_flight_ids_set = set(flight_ids_train)

    flight_id_to_takeoff = {
        d["flight_id"]: d["takeoff"]
        for d in flight_list_lf.select("flight_id", "takeoff").collect().iter_rows(named=True)
    }

    train_trajectories: list[pl.LazyFrame] = []
    validation_trajectories: list[pl.LazyFrame] = []

    all_flight_ids = flight_list_lf.select("flight_id").unique().collect()["flight_id"].to_list()
    for flight_id in track(all_flight_ids, description="processing flights"):
        traj_lf = raw.scan_trajectory(flight_id, partition)
        assert traj_lf is not None

        full_traj_df = traj_lf.sort("timestamp").collect()
        assert full_traj_df.height > 2

        takeoff_ts = flight_id_to_takeoff[flight_id]

        timestamp_s = full_traj_df["timestamp"].dt.epoch(time_unit="ms").to_numpy() / 1000.0
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
                "time_since_takeoff": (full_traj_df["timestamp"] - takeoff_ts).dt.total_seconds(
                    fractional=True
                ),
                "altitude": alt_interp,
                "altitude_is_outlier": alt_outlier_mask,
                "groundspeed": gs_interp,
                "groundspeed_is_outlier": gs_outlier_mask,
                "vertical_rate": vs_interp,
                "vertical_rate_is_outlier": vs_outlier_mask,
            }
        ).with_columns(pl.lit(flight_id).alias("flight_id"))

        if flight_id in train_flight_ids_set:
            train_trajectories.append(processed_traj_df.lazy())
        elif flight_id in flight_ids_validation:
            validation_trajectories.append(processed_traj_df.lazy())

    for split, trajectories in [
        ("train", train_trajectories),
        ("validation", validation_trajectories),
    ]:
        output_path = PATH_PREPROCESSED / f"trajectories_{partition}_{split}.parquet"
        stop_event = multiprocessing.Event()
        monitor_process = multiprocessing.Process(
            target=_monitor_file_size, args=(output_path, stop_event)
        )
        monitor_process.start()
        try:
            pl.concat(trajectories).sink_parquet(output_path)
        finally:
            stop_event.set()
            monitor_process.join()
        logger.info(f"wrote {split} state vectors to {output_path}")


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


#
# data normalisation and splitting
#


class Stat(TypedDict):
    mean: float
    std: float


Stats: TypeAlias = dict[TrajectoryFeature, Stat]


def make_standardisation_stats(
    partition: Partition,
):
    train_traj_lf = pl.scan_parquet(PATH_PREPROCESSED / f"trajectories_{partition}_train.parquet")
    train_flight_ids = train_traj_lf.select("flight_id").unique().collect()["flight_id"].to_list()

    trajectory_iterator = TrajectoryIterator(
        partition=partition,
        split="train",
        flight_ids=train_flight_ids,
        start_to_end_only=True,
    )
    all_segment_dfs = []
    for trajectory in track(trajectory_iterator, description="collecting train segments for stats"):
        segment_features_lf = trajectory.features_df.lazy().select(TRAJECTORY_FEATURES)
        all_segment_dfs.append(segment_features_lf)
    train_segments_lf = pl.concat(all_segment_dfs)

    stats_df = train_segments_lf.select(
        [pl.mean(col).alias(f"{col}_mean") for col in TRAJECTORY_FEATURES]
        + [pl.std(col).alias(f"{col}_std") for col in TRAJECTORY_FEATURES]
    ).collect()

    standardisation_stats: Stats = {}
    row = stats_df.row(0, named=True)
    for feature in TRAJECTORY_FEATURES:
        standardisation_stats[feature] = {
            "mean": row[f"{feature}_mean"],
            "std": row[f"{feature}_std"],
        }

    output_path = PATH_PREPROCESSED / f"stats_{partition}.json"
    with open(output_path, "w") as f:
        json.dump(standardisation_stats, f, indent=2)
    logger.info(f"wrote stats to {output_path}")


#
# wrappers
#


def load_standardisation_stats(
    partition: Partition,
) -> Stats:
    with open(PATH_PREPROCESSED / f"stats_{partition}.json") as f:
        return json.load(f)


class TrajectoryInfo(TypedDict):
    idx: int
    flight_id: str
    start: datetime.datetime
    end: datetime.datetime
    fuel_kg: float
    takeoff: datetime.datetime
    aircraft_type: str


class Trajectory(NamedTuple):
    features_df: pl.DataFrame
    info: TrajectoryInfo


class TrajectoryIterator:
    """Yields the entire flight trajectory for each segment."""

    def __init__(
        self,
        partition: Partition,
        split: Split,
        *,
        flight_ids: Collection[FlightId] | None = None,
        segments_df: pl.DataFrame | None = None,
        shuffle_seed: int | None = None,
        stats: Stats | None = None,
        start_to_end_only: bool = False,
    ):
        if segments_df is None:
            assert flight_ids is not None, "either `flight_ids` or `segments_df` must be provided"
            fuel_lf = raw.scan_fuel(partition)
            flight_list_lf = raw.scan_flight_list(partition)
            segments_df = (
                fuel_lf.filter(pl.col("flight_id").is_in(flight_ids))
                .join(
                    flight_list_lf.select("flight_id", "takeoff", "aircraft_type"), on="flight_id"
                )
                .collect()
            )

        self.segment_infos: list[TrajectoryInfo] = segments_df.to_dicts()  # type: ignore
        traj_lf = pl.scan_parquet(PATH_PREPROCESSED / f"trajectories_{partition}_{split}.parquet")

        if stats is not None:
            standardisation_exprs = [
                ((pl.col(f) - stats[f]["mean"]) / stats[f]["std"]).alias(f)
                for f in TRAJECTORY_FEATURES
            ]
            traj_lf = traj_lf.with_columns(standardisation_exprs)

        all_flight_ids = list({info["flight_id"] for info in self.segment_infos})
        all_trajs_df = traj_lf.filter(pl.col("flight_id").is_in(all_flight_ids)).collect()
        self.traj_cache: dict[FlightId, pl.DataFrame] = {
            flight_id: df for (flight_id,), df in all_trajs_df.group_by("flight_id")
        }

        self.shuffle_seed = shuffle_seed
        self.start_to_end_only = start_to_end_only

    def __len__(self) -> int:
        return len(self.segment_infos)

    def __iter__(self) -> Iterator[Trajectory]:
        indices = np.arange(len(self.segment_infos))
        if self.shuffle_seed is not None:
            rng = np.random.default_rng(self.shuffle_seed)
            rng.shuffle(indices)

        for idx in indices:
            segment_info = self.segment_infos[idx]
            flight_id = segment_info["flight_id"]
            flight_traj_df = self.traj_cache[flight_id]
            if self.start_to_end_only:
                flight_traj_df = flight_traj_df.filter(
                    pl.col("timestamp").is_between(
                        segment_info["start"], segment_info["end"], closed="both"
                    )
                )

            yield Trajectory(
                features_df=flight_traj_df,
                info=segment_info,
            )
