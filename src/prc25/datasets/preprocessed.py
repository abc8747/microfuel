from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from rich.progress import track

from .. import PATH_PREPROCESSED
from . import raw

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Literal

logger = logging.getLogger(__name__)


def _np_interpolate(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    valid_mask = ~np.isnan(y)
    if not np.any(valid_mask):
        return y
    return np.interp(x, x[valid_mask], y[valid_mask])


def create_preprocessed(partition: Literal["train", "rank"]):
    PATH_PREPROCESSED.mkdir(exist_ok=True, parents=True)

    flight_list_lf = raw.scan_flight_list(partition)
    fuel_lf = raw.scan_fuel(partition)

    segments_df = flight_list_lf.join(fuel_lf, on="flight_id").sort("flight_id").collect()
    logger.info(f"found {len(segments_df)} segments in partition `{partition}`")

    processed_trajectories: list[pl.DataFrame] = []

    for (flight_id,), segments in track(
        segments_df.group_by("flight_id"),
        description="processing flights",
    ):
        traj_lf = raw.scan_trajectory(flight_id, partition)
        if traj_lf is None:
            logger.warning(f"skipped flight_id: {flight_id} (no trajectory found)")
            continue

        full_traj_df = traj_lf.sort("timestamp").collect()
        if full_traj_df.height < 2:
            logger.warning(f"skipped flight_id: {flight_id} (insufficient data points)")
            continue

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
        aircraft_type: str = segments.select(pl.col("aircraft_type").first()).item()

        final_df = (
            processed_traj_df.lazy()
            .with_columns(
                pl.lit(flight_id).alias("flight_id"),
                pl.lit(aircraft_type).alias("aircraft_type"),
                segment_expr.alias("segment_id"),
                (
                    (pl.col("timestamp") - takeoff_time)
                    .dt.total_seconds(fractional=True)
                    .alias("time_since_takeoff")
                ),
                pl.col("timestamp").diff().dt.total_seconds(fractional=True).alias("dt"),
            )
            .select(
                "flight_id",
                "aircraft_type",
                "segment_id",
                "time_since_takeoff",
                "dt",
                "altitude",
                "altitude_is_outlier",
                "groundspeed",
                "groundspeed_is_outlier",
                "vertical_rate",
                "vertical_rate_is_outlier",
            )
            .collect()
        )
        processed_trajectories.append(final_df)

    all_trajectories_df = pl.concat(processed_trajectories)

    output_path = PATH_PREPROCESSED / f"trajectories_{partition}.parquet"
    all_trajectories_df.write_parquet(output_path)
    logger.info(f"wrote {len(all_trajectories_df)} combined state vectors to {output_path}")
