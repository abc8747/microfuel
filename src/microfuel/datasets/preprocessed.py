from __future__ import annotations

import gc
import json
import logging
import multiprocessing
import traceback
from collections import namedtuple
from datetime import timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict, get_args

import numba as nb
import numpy as np
import polars as pl
from rich.progress import track

from .. import (
    PATH_DATA_RAW,
    PATH_PREPROCESSED,
    Coordinate2D,
    Partition,
    Split,
    deg2rad,
    fpm2mps,
    ft2m,
    knot2mps,
)
from . import raw

if TYPE_CHECKING:
    from datetime import datetime
    from multiprocessing.synchronize import Event
    from typing import Annotated, Collection, Iterator, TypeAlias

    import isqx
    import isqx.aerospace

    from .. import AirportIcao, FlightId, SegmentId, Split


logger = logging.getLogger(__name__)

CoreFeature = Literal[
    "altitude",
    "groundspeed",
    "vertical_rate",
    # "track_rate"
]
# NOTE: weather data is not used and reserved for future use.
# WeatherFeature = Literal[
#     "wind_dot_ground",
#     # "true_airspeed"
# ]
# WEATHER_FEATURES = get_args(WeatherFeature)
WEATHER_FEATURES = []
StateFeature = Literal[CoreFeature]
STATE_FEATURES = get_args(StateFeature)
FlightFeature = Literal["flight_progress", "flight_duration"]
FLIGHT_FEATURES = get_args(FlightFeature)

MODEL_INPUT_FEATURES: list[str] = [
    *FLIGHT_FEATURES,
    *STATE_FEATURES,
]


def make_splits(
    partition: Partition,
    train_split: float = 0.8,
    seed: int = 13,
    *,
    path_base: Path = PATH_PREPROCESSED,
    max_bins: int = 30,
    min_samples_for_binning: int = 2,
):  # TODO: allow k fold stratified splits
    path_base.mkdir(exist_ok=True, parents=True)
    flight_list_lf = raw.scan_flight_list(partition)
    fuel_lf = raw.scan_fuel(partition)

    segments_df = (
        fuel_lf.with_columns(
            (pl.col("end") - pl.col("start")).dt.total_seconds().alias("duration_s")
        )
        .join(flight_list_lf.select("flight_id", "aircraft_type"), on="flight_id")
        .select("idx", "aircraft_type", "duration_s")
        .sort("idx")
        .collect()
    )
    logger.info(f"found {len(segments_df)} segments with fuel data in `{partition}`")

    bin_boundaries_data = []

    def add_duration_bin(group_df: pl.DataFrame) -> pl.DataFrame:
        ac_type = group_df["aircraft_type"][0]
        duration_series = group_df["duration_s"]
        n_samples = duration_series.len()
        assert n_samples > 0

        duration_quantiles = (
            min(max_bins, int(1 + np.log2(n_samples)))
            if n_samples >= min_samples_for_binning
            else 1
        )

        quantile_points = np.linspace(0, 1, duration_quantiles + 1)
        breaks_set: set[float] = set()
        for q in quantile_points:
            b = duration_series.quantile(q, interpolation="linear")
            assert b is not None
            breaks_set.add(b)
        breaks = sorted(breaks_set)

        if len(breaks) < 2:
            min_val = float(duration_series.min() or 0)  # type: ignore
            max_val = float(duration_series.max() or 1)  # type: ignore
            breaks = [min_val, max_val] if min_val != max_val else [min_val, min_val + 1]

        # last break should be inclusive of the max value
        max_dur = float(duration_series.max())  # type: ignore
        if max_dur is not None and breaks[-1] < max_dur:
            breaks[-1] = max_dur

        labels = [f"d_q{i}" for i in range(len(breaks) - 1)]
        for i, label in enumerate(labels):
            bin_boundaries_data.append(
                {
                    "aircraft_type": ac_type,
                    "duration_bin": label,
                    "lower_bound": breaks[i],
                    "upper_bound": breaks[i + 1],
                }
            )

        bin_expr = pl.when(pl.col("duration_s") <= breaks[1]).then(pl.lit(labels[0]))
        for i in range(2, len(breaks) - 1):
            bin_expr = bin_expr.when(pl.col("duration_s") <= breaks[i]).then(pl.lit(labels[i - 1]))
        bin_expr = bin_expr.otherwise(pl.lit(labels[-1]))

        return group_df.with_columns(bin_expr.alias("duration_bin"))

    segments_binned_df = segments_df.group_by("aircraft_type", maintain_order=True).map_groups(
        add_duration_bin
    )
    bin_boundaries_df = pl.DataFrame(bin_boundaries_data)

    stratify_cols = ["aircraft_type", "duration_bin"]
    n_train_samples_expr = pl.max_horizontal(1, (pl.count() * train_split).floor())
    train_df = segments_binned_df.filter(
        pl.int_range(0, pl.count()).shuffle(seed=seed).over(stratify_cols)
        < n_train_samples_expr.over(stratify_cols)
    )

    train_segment_ids_set = set(train_df["idx"].to_list())
    all_segment_ids_set = set(segments_binned_df["idx"].to_list())
    validation_segment_ids_set = all_segment_ids_set - train_segment_ids_set

    segment_ids_train = sorted(list(train_segment_ids_set))
    segment_ids_validation = sorted(list(validation_segment_ids_set))

    logger.info(
        f"stratified split by {stratify_cols}: {len(segment_ids_train)} train, {len(segment_ids_validation)} validation"
    )

    train_counts_df = train_df.group_by(stratify_cols).len().rename({"len": "train_count"})
    validation_df = segments_binned_df.filter(pl.col("idx").is_in(validation_segment_ids_set))
    validation_counts_df = (
        validation_df.group_by(stratify_cols).len().rename({"len": "validation_count"})
    )

    all_groups_df = segments_binned_df.select(stratify_cols).unique().sort(stratify_cols)
    combined_counts_df = (
        all_groups_df.join(train_counts_df, on=stratify_cols, how="left")
        .join(validation_counts_df, on=stratify_cols, how="left")
        .fill_null(0)
    )

    logging_df = combined_counts_df.join(bin_boundaries_df, on=stratify_cols, how="left")

    logger.info("split counts by stratification groups:")
    _ac_types: set[str] = set()
    for row in logging_df.sort(["aircraft_type", "duration_bin"]).iter_rows(named=True):
        ac_type = t if (t := row["aircraft_type"]) not in _ac_types else ""
        _ac_types.add(row["aircraft_type"])
        lower = row["lower_bound"]
        upper = row["upper_bound"]
        train_count = row["train_count"]
        validation_count = row["validation_count"]
        total = train_count + validation_count
        train_pct = train_count / total if total > 0 else 0
        duration_str = f"({lower or 0:.0f}s-{upper or 0:.0f}s]"
        logger.info(
            f"  {ac_type:<5}{duration_str:<14}: {train_count:>5}/{validation_count:>5} ({train_pct:5.1%})"
        )

    splits: dict[Split, list[SegmentId]] = {
        "train": segment_ids_train,
        "validation": segment_ids_validation,
    }
    output_path = path_base / f"splits_{partition}.json"
    with open(output_path, "w") as f:
        json.dump(splits, f)
    logger.info(f"wrote splits to {output_path}")


def find_segment_indices(timestamps, start_time, end_time, *, xp=np):
    start_idx = xp.searchsorted(timestamps, start_time, side="left")
    end_idx = xp.searchsorted(timestamps, end_time, side="right")
    return start_idx, end_idx


@nb.njit(cache=True)
def _kalman_filter(
    measurements: np.ndarray,
    dts: np.ndarray,
    initial_state_mean: np.ndarray,
    initial_state_covariance: np.ndarray,
    transition_matrix_fn_val: np.ndarray,
    observation_matrix: np.ndarray,
    process_noise_covariance: np.ndarray,
    observation_noise_covariance: np.ndarray,
):
    n_timesteps = measurements.shape[0]
    n_dim_state = initial_state_mean.shape[0]

    filtered_state_means = np.zeros((n_timesteps, n_dim_state))
    filtered_state_covariances = np.zeros((n_timesteps, n_dim_state, n_dim_state))

    x = initial_state_mean.copy()
    p = initial_state_covariance.copy()

    filtered_state_means[0] = x
    filtered_state_covariances[0] = p
    identity_matrix = np.eye(n_dim_state)

    for t in range(1, n_timesteps):
        f = transition_matrix_fn_val.copy()
        f[0, 1] = dts[t - 1]

        x_pred = f @ x
        p_pred = f @ p @ f.T + process_noise_covariance

        z = measurements[t]
        is_nan = np.isnan(z)

        if is_nan:
            x = x_pred
            p = p_pred
        else:
            h = observation_matrix
            y = z - h @ x_pred
            s = h @ p_pred @ h.T + observation_noise_covariance
            s_inv = np.linalg.inv(s)
            k = p_pred @ h.T @ s_inv
            x = x_pred + k @ y
            p = (identity_matrix - k @ h) @ p_pred

        filtered_state_means[t] = x
        filtered_state_covariances[t] = p

    return filtered_state_means, filtered_state_covariances


@nb.njit(cache=True)
def _rts_smoother_numba(
    filtered_state_means: np.ndarray,
    filtered_state_covariances: np.ndarray,
    dts: np.ndarray,
    transition_matrix_fn_val: np.ndarray,
    process_noise_covariance: np.ndarray,
):
    n_timesteps, n_dim_state = filtered_state_means.shape
    smoothed_state_means = filtered_state_means.copy()
    smoothed_state_covariances = filtered_state_covariances.copy()

    for k in range(n_timesteps - 2, -1, -1):
        x_k_k = filtered_state_means[k]
        p_k_k = filtered_state_covariances[k]
        x_k1_n = smoothed_state_means[k + 1]
        p_k1_n = smoothed_state_covariances[k + 1]

        f = transition_matrix_fn_val.copy()
        f[0, 1] = dts[k]
        q = process_noise_covariance

        x_k1_k = f @ x_k_k
        p_k1_k = f @ p_k_k @ f.T + q
        p_k1_k_inv = np.linalg.inv(p_k1_k)

        c_k = p_k_k @ f.T @ p_k1_k_inv

        smoothed_state_means[k] = x_k_k + c_k @ (x_k1_n - x_k1_k)
        smoothed_state_covariances[k] = p_k_k + c_k @ (p_k1_n - p_k1_k) @ c_k.T

    return smoothed_state_means, smoothed_state_covariances


SmoothResult = namedtuple("SmoothResult", ["val", "val_d", "var_val", "var_val_d"])


def smooth_time_series(
    values,
    dts_s,
    process_noise_variances: tuple[float, float],
    observation_noise_variance: float,
    gap_threshold: float = 30.0,
) -> SmoothResult:
    """Applies a Kalman filter and RTS smoother to a 1D time series, handling large gaps.

    Assumes the time series follow a Constant Velocity (CV) system:
    $x_k = F x_{k-1} + w_k$.

    :param process_noise_variances: (pos, vel) variances for the model's state transition noise (Q).
    :param observation_noise_variance: variance for the measurement noise (R).
    :param gap_threshold: time gap (in seconds) above which to split the time series into chunks.
    """
    gap_indices = np.where(dts_s > gap_threshold)[0]
    chunk_boundaries = np.concatenate(([0], gap_indices + 1, [len(values)]))

    all_smoothed_values = np.full_like(values, np.nan)
    all_smoothed_derivatives = np.full_like(values, np.nan)
    all_smoothed_value_variances = np.full_like(values, np.nan)
    all_smoothed_derivative_variances = np.full_like(values, np.nan)

    transition_matrix_template = np.array([[1.0, 0.0], [0.0, 1.0]])
    observation_matrix = np.array([[1.0, 0.0]])
    process_noise_covariance = np.diag(np.array(process_noise_variances, dtype=np.float64))
    observation_noise_covariance = np.array([[observation_noise_variance]])

    for i in range(len(chunk_boundaries) - 1):
        start, end = chunk_boundaries[i], chunk_boundaries[i + 1]
        chunk_values = values[start:end]
        chunk_dts = dts_s[start : end - 1]

        if len(chunk_values) < 2:
            continue

        first_valid_idx = np.where(~np.isnan(chunk_values))[0]
        if len(first_valid_idx) == 0:
            continue
        initial_value = chunk_values[first_valid_idx[0]]
        initial_state_mean = np.array([initial_value, 0.0])
        initial_state_covariance = np.eye(2) * 1e5

        filtered_means, filtered_covs = _kalman_filter(
            measurements=chunk_values,
            dts=chunk_dts,
            initial_state_mean=initial_state_mean,
            initial_state_covariance=initial_state_covariance,
            transition_matrix_fn_val=transition_matrix_template,
            observation_matrix=observation_matrix,
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
        )

        smoothed_means, smoothed_covs = _rts_smoother_numba(
            filtered_means,
            filtered_covs,
            chunk_dts,
            transition_matrix_template,
            process_noise_covariance,
        )

        all_smoothed_values[start:end] = smoothed_means[:, 0]
        all_smoothed_derivatives[start:end] = smoothed_means[:, 1]
        all_smoothed_value_variances[start:end] = smoothed_covs[:, 0, 0]
        all_smoothed_derivative_variances[start:end] = smoothed_covs[:, 1, 1]

    return SmoothResult(
        all_smoothed_values,
        all_smoothed_derivatives,
        all_smoothed_value_variances,
        all_smoothed_derivative_variances,
    )


def _np_interpolate(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    valid_mask = ~np.isnan(y)
    if not np.any(valid_mask):
        return y
    return np.interp(x, x[valid_mask], y[valid_mask])


def make_trajectories(
    partition: Partition,
    seed: int = 13,
    *,
    path_base: Path = PATH_PREPROCESSED,
    altitude_max: Annotated[float, isqx.aerospace.PRESSURE_ALTITUDE(isqx.M)] = ft2m(50000),
    speed_max: Annotated[float, isqx.SPEED(isqx.M_PERS)] = knot2mps(800),
    vertical_speed_max: Annotated[float, isqx.aerospace.VERTICAL_RATE(isqx.M_PERS)] = fpm2mps(8000),
    track_rate_max: Annotated[float, isqx.RAD_PERS] = 0.003,
    plot_every_n_flights: int | None = None,
):
    """Creates train/validation split of preprocessed trajectories.

    Handles the alignment of asynchronous data sources:

    1. Flight List: [takeoff, landing] constraints.
    2. Fuel Data: segment boundaries.
    3. ADS-B + ACARS: raw state observations.

    It produces the standard state vector $x_t$ required by
    [`microfuel.model.FuelBurnPredictor`][].

    Everything related to segments (e.g. whether a particular state vector is within [start, end])
    should be handled elsewhere. This function processes the *entire* trajectory.
    """
    path_base.mkdir(exist_ok=True, parents=True)

    flight_list_lf = raw.scan_flight_list(partition)
    fuel_lf = raw.scan_fuel(partition)

    flight_ids_with_fuel = (
        flight_list_lf.join(fuel_lf, on="flight_id")
        .select("flight_id")
        .unique()
        .sort("flight_id")
        .collect()
        .to_series()
        .shuffle(seed=seed)
        .to_list()
    )
    logger.info(
        f"found {len(flight_ids_with_fuel)} flights with fuel data in partition `{partition}`"
    )

    flight_id_to_flight: dict[FlightId, raw.FlightListRecord] = {
        row["flight_id"]: row  # type: ignore
        for row in flight_list_lf.collect().iter_rows(named=True)
    }
    flight_id_to_segment: dict[FlightId, tuple[pl.Series, pl.Series]] = {
        flight_id: (data["start"], data["end"])
        for (flight_id,), data in fuel_lf.collect().group_by("flight_id")
    }
    icao_to_coords: dict[AirportIcao, Coordinate2D[float]] = {
        row["icao"]: Coordinate2D(lng=row["longitude"], lat=row["latitude"])
        for row in raw.scan_airports().collect().iter_rows(named=True)
    }  # type: ignore

    trajectories_all: list[pl.LazyFrame] = []
    for i, flight_id in enumerate(track(flight_ids_with_fuel, description="processing flights")):
        traj_lf = raw.scan_trajectory(flight_id, partition).with_columns(
            pl.col("timestamp").dt.replace_time_zone("UTC")
        )  # raw file has naiive timestamps, cast early to avoid issues in era5 interpolation
        flight = flight_id_to_flight[flight_id]
        timestamp_takeoff = flight["takeoff"]
        timestamp_landed = flight["landed"]
        ac_type = flight["aircraft_type"]

        # segments can cover timestamps that are missing from the trajectory data, including
        # timestamps that start before takeoff or end after landing.
        # so we want to make sure one segment has at least 2 points (start and end) present
        timestamps_segment_start, timestamps_segment_end = flight_id_to_segment[flight_id]
        timestamps_required = (
            pl.concat(
                (
                    pl.Series((timestamp_takeoff, timestamp_landed)).dt.cast_time_unit("ns"),
                    timestamps_segment_start,
                    timestamps_segment_end,
                )
            )
            .unique()
            .sort()
        )

        # NOTE: discarding duplicate timestamps is a bad idea! sometimes the time gets "stuck"
        # and we lose a lot of useful information.
        traj_df = traj_lf.unique(subset=["timestamp"], keep="first").sort("timestamp").collect()
        timestamps_existing = traj_df.select("timestamp").to_series()
        # takeoff time in flight list usually precedes the first timestamp in trajectory data
        timestamps_missing = timestamps_required.filter(
            ~timestamps_required.is_in(timestamps_existing)
        )

        if timestamps_missing.len() > 0:
            # for required timestamps that are beyond what the trajectory data provides,
            # we assume they are stationary on the ground, zero filling features
            timestamp_gnd_start: datetime = min(timestamp_takeoff, timestamps_existing.min())  # type: ignore
            timestamp_gnd_end: datetime = max(timestamp_landed, timestamps_existing.max())  # type: ignore
            coord_origin, coord_dest = (
                icao_to_coords[flight["origin_icao"]],
                icao_to_coords[flight["destination_icao"]],
            )  # we dont need elevation since altitude is barometric
            trks: list[float] = traj_df.select("track").drop_nulls().to_series().to_list()
            trk_start, trk_end = (trks[0], trks[-1]) if len(trks) else (0.0, 0.0)  # ffill/bfill

            def _artificial(ts: datetime) -> raw.TrajectoryRecord:
                if ts <= timestamp_gnd_start:
                    val, lng, lat, trk = 0.0, coord_origin.lng, coord_origin.lat, trk_start
                elif ts >= timestamp_gnd_end:
                    val, lng, lat, trk = 0.0, coord_dest.lng, coord_dest.lat, trk_end
                else:
                    val, lng, lat, trk = None, None, None, None
                return raw.TrajectoryRecord(
                    timestamp=ts,
                    flight_id=flight_id,
                    typecode=ac_type,
                    latitude=lat,
                    longitude=lng,
                    altitude=val,
                    groundspeed=val,
                    track=trk,
                    vertical_rate=val,
                    mach=val,
                    TAS=val,
                    CAS=val,
                    source="artificial",
                )

            artificial_df = pl.DataFrame(_artificial(ts) for ts in timestamps_missing).with_columns(
                pl.col("timestamp").dt.cast_time_unit("ns")
            )
            full_traj_df = traj_df.vstack(artificial_df).sort("timestamp")
        else:
            full_traj_df = traj_df.sort("timestamp")

        timestamp_s = full_traj_df["timestamp"].dt.epoch(time_unit="ms").to_numpy() / 1000.0
        vs_raw = fpm2mps(full_traj_df["vertical_rate"].to_numpy())
        alt_raw = ft2m(full_traj_df["altitude"].to_numpy())
        gs_raw = knot2mps(full_traj_df["groundspeed"].to_numpy())
        track_raw_rad = np.deg2rad(full_traj_df["track"].to_numpy())
        lat_raw = full_traj_df["latitude"].to_numpy()
        lng_raw = full_traj_df["longitude"].to_numpy()

        vs_outlier_mask = (np.abs(vs_raw) > vertical_speed_max) | np.isnan(vs_raw)
        alt_outlier_mask = (alt_raw > altitude_max) | np.isnan(alt_raw)
        gs_outlier_mask = (gs_raw > speed_max) | np.isnan(gs_raw)
        track_outlier_mask = np.isnan(track_raw_rad)

        vs_with_nan = np.where(vs_outlier_mask, np.nan, vs_raw)
        alt_with_nan = np.where(alt_outlier_mask, np.nan, alt_raw)

        v_east_raw = gs_raw * np.sin(track_raw_rad)
        v_north_raw = gs_raw * np.cos(track_raw_rad)
        v_east_with_nan = np.where(gs_outlier_mask | track_outlier_mask, np.nan, v_east_raw)
        v_north_with_nan = np.where(gs_outlier_mask | track_outlier_mask, np.nan, v_north_raw)

        dts_s = np.diff(timestamp_s)
        alt_res = smooth_time_series(
            values=alt_with_nan,
            dts_s=dts_s,
            process_noise_variances=(1.0**2, 0.3**2),
            observation_noise_variance=4.0**2,
        )
        vs_res = smooth_time_series(
            values=vs_with_nan,
            dts_s=dts_s,
            process_noise_variances=(0.3**2, 0.1**2),
            observation_noise_variance=1.0**2,
        )
        v_east_res = smooth_time_series(
            values=v_east_with_nan,
            dts_s=dts_s,
            process_noise_variances=(1.0**2, 0.1**2),
            observation_noise_variance=6.0**2,
        )
        v_north_res = smooth_time_series(
            values=v_north_with_nan,
            dts_s=dts_s,
            process_noise_variances=(1.0**2, 0.1**2),
            observation_noise_variance=6.0**2,
        )

        v_east_smooth, v_east_dot_smooth = v_east_res.val, v_east_res.val_d
        v_north_smooth, v_north_dot_smooth = v_north_res.val, v_north_res.val_d
        gs_smooth = np.sqrt(v_east_smooth**2 + v_north_smooth**2)
        gs_smooth_outlier_mask = (gs_smooth > speed_max) | (gs_smooth < 0.0)
        track_rate_smooth = np.abs(
            (v_north_smooth * v_east_dot_smooth - v_east_smooth * v_north_dot_smooth)
            / np.clip(v_east_smooth**2 + v_north_smooth**2, 1e-6, None)
        )
        track_rate_outlier_mask = track_rate_smooth > track_rate_max
        gs_track_outlier_mask = gs_smooth_outlier_mask | track_rate_outlier_mask
        # ground speed and track rate are derived from ve and vn if either fails, set as outlier.
        gs_smooth[gs_track_outlier_mask] = np.nan
        track_rate_smooth[gs_track_outlier_mask] = np.nan

        v_east_interp = _np_interpolate(v_east_smooth, timestamp_s)
        v_north_interp = _np_interpolate(v_north_smooth, timestamp_s)

        # 0=N, 90=E
        track_interp_rad = np.arctan2(v_east_interp, v_north_interp)
        track_interp_deg = np.rad2deg(track_interp_rad)
        track_interp_deg = np.where(track_interp_deg < 0, track_interp_deg + 360, track_interp_deg)

        if i < 100 or (plot_every_n_flights is not None and i % plot_every_n_flights == 0):
            # import matplotlib
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec

            # matplotlib.use("WebAgg")
            from .. import PATH_PLOTS_OUTPUT

            N_PLOTS = 4
            fig = plt.figure(figsize=(9, 9 * N_PLOTS * 0.3), layout="tight")
            gs = GridSpec(N_PLOTS, 1, figure=fig)

            ax_alt = fig.add_subplot(gs[0])
            ax_vs = fig.add_subplot(gs[1], sharex=ax_alt)
            ax_gs = fig.add_subplot(gs[2], sharex=ax_alt)
            ax_track = fig.add_subplot(gs[3], sharex=ax_alt)

            for ax in [ax_alt, ax_vs, ax_gs, ax_track]:
                if ax != ax_track:
                    plt.setp(ax.get_xticklabels(), visible=False)
                ax.axvline(timestamp_takeoff.timestamp(), color="green", linewidth=0.5)
                ax.axvline(timestamp_landed.timestamp(), color="blue", linewidth=0.5)
                for j, (start_ts, end_ts) in enumerate(
                    zip(timestamps_segment_start, timestamps_segment_end)
                ):
                    ax.axvspan(start_ts.timestamp(), end_ts.timestamp(), color=f"C{j}", alpha=0.1)
                ax.grid(True, linewidth=0.2)

            ax_alt.plot(timestamp_s, alt_raw, "k.", markersize=2, alpha=0.3, label="raw altitude")
            ax_alt.plot(timestamp_s, alt_res.val, "r-", linewidth=0.5, label="smoothed altitude")
            alt_std = np.sqrt(alt_res.var_val)
            ax_alt.fill_between(
                timestamp_s,
                alt_res.val - alt_std,
                alt_res.val + alt_std,
                color="r",
                alpha=0.2,
                label=r"$\pm 1 \sigma$",
            )
            ax_alt.set_ylabel("altitude (m)")
            ax_alt.set_ylim(0, altitude_max)
            ax_alt.legend()

            ax_vs.plot(
                timestamp_s, vs_raw, "k.", markersize=2, alpha=0.3, label="raw vertical rate"
            )
            ax_vs.plot(
                timestamp_s,
                vs_res.val,
                "r-",
                linewidth=0.5,
                label="smoothed vertical rate",
            )
            ax_vs.plot(
                timestamp_s,
                alt_res.val_d,
                "b--",
                linewidth=0.5,
                label="smoothed altitude derivative",
            )
            vs_std = np.sqrt(vs_res.var_val)
            ax_vs.fill_between(
                timestamp_s,
                vs_res.val - vs_std,
                vs_res.val + vs_std,
                color="r",
                alpha=0.2,
                label=r"$\pm 1 \sigma$ (vs)",
            )
            ax_vs.set_ylabel("vertical rate (m/s)")
            ax_vs.set_ylim(-vertical_speed_max, vertical_speed_max)
            ax_vs.legend()

            ax_gs.plot(timestamp_s, gs_raw, "k.", markersize=2, alpha=0.3, label="raw groundspeed")
            ax_gs.plot(
                timestamp_s,
                gs_smooth,
                "r-",
                linewidth=0.5,
                label="smoothed groundspeed",
            )
            ax_gs.set_ylabel("groundspeed (m/s)")
            ax_gs.set_ylim(0, speed_max)
            ax_gs.legend()

            ax_track.plot(
                timestamp_s, track_interp_deg, "r.", markersize=0.5, label="smoothed track"
            )
            ax_track.set_ylabel("track (deg)")
            ax_track.set_ylim(0, 360)
            ax_track.legend(loc="upper left")
            ax_track_rate = ax_track.twinx()
            ax_track_rate.plot(
                timestamp_s,
                np.rad2deg(track_rate_smooth),
                "b.",
                markersize=2,
                label="track rate",
            )
            ax_track_rate.set_ylabel("track rate (deg/s)", color="b")
            ax_track_rate.tick_params(axis="y", labelcolor="b")
            ax_track_rate.legend(loc="lower right")
            ax_track_rate.set_ylim(-np.rad2deg(track_rate_max), np.rad2deg(track_rate_max))

            ax_track.set_xlabel("time (s)")
            fig.suptitle(f"flight id: {flight_id}")

            # plt.show()
            output_dir = PATH_PLOTS_OUTPUT / "preprocessed_trajectories"
            output_dir.mkdir(exist_ok=True, parents=True)
            output_path = output_dir / f"{partition}_{flight_id}.png"
            fig.savefig(output_path, dpi=300)
            plt.close(fig)

        # its possible that the very start of the smoothed data isn't processed
        # so we must interpolate.
        processed_traj_df = pl.DataFrame(
            {
                "timestamp": full_traj_df["timestamp"],
                "time_since_takeoff": (
                    full_traj_df["timestamp"] - timestamp_takeoff
                ).dt.total_seconds(fractional=True),
                "time_till_arrival": (
                    timestamp_landed - full_traj_df["timestamp"]
                ).dt.total_seconds(fractional=True),
                "latitude": _np_interpolate(lat_raw, timestamp_s),
                "longitude": _np_interpolate(lng_raw, timestamp_s),
                "vertical_rate": _np_interpolate(vs_res.val, timestamp_s),
                "vertical_rate_is_outlier": (vs_outlier_mask | np.isnan(vs_res.val)),
                "altitude": _np_interpolate(alt_res.val, timestamp_s),
                "altitude_is_outlier": (alt_outlier_mask | np.isnan(alt_res.val)),
                "groundspeed": _np_interpolate(gs_smooth, timestamp_s),
                "groundspeed_is_outlier": (
                    gs_outlier_mask | gs_smooth_outlier_mask | np.isnan(gs_smooth)
                ),
                "track": track_interp_deg,
                "track_rate": _np_interpolate(track_rate_smooth, timestamp_s),
                "track_rate_is_outlier": (track_rate_outlier_mask | np.isnan(track_rate_smooth)),
            }
        ).with_columns(pl.lit(flight_id).alias("flight_id"))

        trajectories_all.append(processed_traj_df.lazy())

    output_path = path_base / f"trajectories_{partition}.parquet"
    stop_event = multiprocessing.Event()
    monitor_process = multiprocessing.Process(
        target=_monitor_file_size, args=(output_path, stop_event)
    )
    monitor_process.start()
    try:
        pl.concat(trajectories_all).sink_parquet(output_path)
    finally:
        stop_event.set()
        monitor_process.join()
    logger.info(f"wrote state vectors to {output_path}")


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


def altitude_to_pressure_std(altitude_m):
    # P ~ P0 * (1 - L*h/T0)^(gM/RL)
    return 1013.25 * (1 - 2.25577e-5 * altitude_m).pow(5.25588)


def make_era5(
    partition: Partition,
    *,
    path_base: Path = PATH_PREPROCESSED,
    path_raw_weather: Path = PATH_DATA_RAW / "era5",
):
    import xarray as xr

    path_base.mkdir(exist_ok=True, parents=True)
    path_out_dir = path_base / f"weather_{partition}"
    path_out_dir.mkdir(exist_ok=True, parents=True)

    traj_lf = pl.scan_parquet(path_base / f"trajectories_{partition}.parquet")

    df_coords = (
        traj_lf.select(
            "flight_id",
            "timestamp",
            "latitude",
            "longitude",
            "altitude",
        )
        .with_columns(
            date_key=pl.col("timestamp").dt.convert_time_zone("UTC").dt.date(),
            pressure_level=altitude_to_pressure_std(pl.col("altitude")),
            longitude_era5=(pl.col("longitude") + 360) % 360,
        )
        .sort("timestamp")
        .collect()
    )

    unique_dates = df_coords["date_key"].unique().sort()
    logger.info(f"extracting weather for {len(unique_dates)} unique days")

    def _process_variable(
        variable_dir: Path,
        variable_names: list[str],
        batch_times: tuple,
        targets: dict,
    ) -> np.ndarray | None:
        files = sorted(variable_dir.glob("*.nc"), key=lambda p: int(p.stem))
        if not files:
            return None

        levels = [int(p.stem) for p in files]

        try:
            ds = xr.open_mfdataset(
                files,
                combine="nested",
                concat_dim="level",
                parallel=False,
                chunks={"time": 1},
            )
            ds.coords["level"] = levels

            var_name = next((v for v in variable_names if v in ds), None)
            if not var_name:
                raise ValueError(f"vars {variable_names} not found in {variable_dir}")

            ds = ds.sortby(["time", "latitude", "level"])

            min_t, max_t = batch_times
            da_sliced = ds[var_name].sel(time=slice(min_t, max_t))

            da_loaded = da_sliced.load()
            ds.close()

            # NOTE: fill_value enables extrapolation.
            # consider a point at 23:55,
            # we would need 23:00 and 00:00 (the latter is located in a
            # different file) to interpolate. however, we cannot afford spending
            # 2x RAM, so we just have to deal with it for now.
            interp_res = da_loaded.interp(
                time=targets["time"],
                latitude=targets["latitude"],
                longitude=targets["longitude"],
                level=targets["level"],
                method="linear",
                kwargs={"bounds_error": False, "fill_value": None},
            )

            result = interp_res.values

            del da_loaded
            del ds
            gc.collect()

            return result

        except Exception:
            logger.error(f"failed to process {variable_dir}:\n{traceback.format_exc()}")
            return None

    for date_key in track(unique_dates, description="processing daily weather"):
        output_path = path_out_dir / f"{date_key}.parquet"
        if output_path.exists():
            continue

        batch = df_coords.filter(pl.col("date_key") == date_key)
        if batch.height == 0:
            continue

        logger.info(f"processing {date_key}: {batch.height} points")

        year = f"{date_key.year}"
        month = f"{date_key.month:02d}"
        day = f"{date_key.day:02d}"
        day_path = path_raw_weather / year / month / day

        if not day_path.exists():
            continue

        target_time = xr.DataArray(batch["timestamp"].to_numpy(), dims="points")
        target_lats = xr.DataArray(batch["latitude"].to_numpy(), dims="points")
        target_lons = xr.DataArray(batch["longitude_era5"].to_numpy(), dims="points")
        target_level = xr.DataArray(batch["pressure_level"].to_numpy(), dims="points")

        targets = {
            "time": target_time,
            "latitude": target_lats,
            "longitude": target_lons,
            "level": target_level,
        }

        min_time = batch["timestamp"].min().astimezone(timezone.utc).replace(  # type: ignore
            tzinfo=None
        ) - timedelta(hours=2)
        max_time = batch["timestamp"].max().astimezone(timezone.utc).replace(  # type: ignore
            tzinfo=None
        ) + timedelta(hours=2)

        u_values = _process_variable(
            day_path / "u_component_of_wind",
            ["u", "u_component_of_wind", "var131"],
            (min_time, max_time),
            targets,
        )
        if u_values is None:
            continue

        v_values = _process_variable(
            day_path / "v_component_of_wind",
            ["v", "v_component_of_wind", "var132"],
            (min_time, max_time),
            targets,
        )
        if v_values is None:
            continue

        chunk_res = pl.DataFrame(
            {
                "flight_id": batch["flight_id"],
                "timestamp": batch["timestamp"],
                "u_wind": u_values,
                "v_wind": v_values,
            }
        )
        chunk_res.write_parquet(output_path)

        del chunk_res
        del u_values
        del v_values
        gc.collect()

    logger.info(f"wrote daily weather chunks to {path_out_dir}")


def make_derived_features(partition: Partition, *, path_base: Path = PATH_PREPROCESSED):
    """
    !!! warning
        This function is unused. Integration of weather features is planned for the future
    """
    traj_lf = (
        pl.scan_parquet(path_base / f"trajectories_{partition}.parquet")
        .select("flight_id", "timestamp", "groundspeed", "track")
        .with_row_index("row_idx")
    )
    weather_lf = (
        pl.scan_parquet(path_base / f"weather_{partition}/*.parquet")
        .select("flight_id", "timestamp", "u_wind", "v_wind")
        .unique(subset=["flight_id", "timestamp"])
    )

    ve = pl.col("groundspeed") * (deg2rad(pl.col("track"))).sin()
    vn = pl.col("groundspeed") * (deg2rad(pl.col("track"))).cos()

    u, v = pl.col("u_wind"), pl.col("v_wind")

    va_e = ve - u
    va_n = vn - v

    tas = (va_e.pow(2) + va_n.pow(2)).sqrt()
    wind_dot = ve * u + vn * v

    derived_lf = (
        traj_lf.join(weather_lf, on=["flight_id", "timestamp"], how="left")
        .sort("row_idx")
        .select(
            "flight_id",
            "timestamp",
            tas.alias("true_airspeed"),
            wind_dot.alias("wind_dot_ground"),
        )
    )

    path_out = path_base / f"derived_{partition}.parquet"
    derived_lf.sink_parquet(path_out)
    logger.info(f"wrote derived features to {path_out}")


#
# data normalisation and splitting
#


class Stat(TypedDict):
    mean: float
    std: float


Stats: TypeAlias = dict[StateFeature | FlightFeature, Stat]


def make_standardisation_stats(
    partition: Partition,
    *,
    path_base: Path = PATH_PREPROCESSED,
):
    splits = load_splits(partition, path_base=path_base)
    train_segment_ids = splits["train"]
    logger.info(f"computing standardisation stats from {len(train_segment_ids)} train segments")

    fuel_lf = raw.scan_fuel(partition).filter(pl.col("idx").is_in(train_segment_ids))
    train_flight_ids_df = fuel_lf.select("flight_id").unique().collect()
    flight_list_df = (
        raw.scan_flight_list(partition)
        .filter(pl.col("flight_id").is_in(train_flight_ids_df["flight_id"]))
        .collect()
    )

    flight_duration_s = (flight_list_df["landed"] - flight_list_df["takeoff"]).dt.total_seconds(
        fractional=True
    )
    standardisation_stats: Stats = {
        "flight_duration": {
            "mean": flight_duration_s.mean(),
            "std": flight_duration_s.std(),
        }
    }  # type: ignore

    trajectory_iterator = TrajectoryIterator(
        partition=partition,
        segment_ids=train_segment_ids,
        start_to_end_only=True,
    )

    features_to_stat = [*STATE_FEATURES, "flight_progress"]
    running_stats = {
        feature: {"sum": 0.0, "sum_sq": 0.0, "count": 0} for feature in features_to_stat
    }

    flight_id_to_duration = {
        row["flight_id"]: (row["landed"] - row["takeoff"]).total_seconds()
        for row in flight_list_df.iter_rows(named=True)
    }

    for trajectory in track(trajectory_iterator, description="computing stats from train segments"):
        segment_df = trajectory.features_df
        count = len(segment_df)
        if count == 0:
            continue

        duration_s = flight_id_to_duration.get(trajectory.info["flight_id"])
        if duration_s is not None and duration_s > 0:
            progress = (segment_df["timestamp"] - trajectory.info["takeoff"]).dt.total_seconds(
                fractional=True
            ) / duration_s
            running_stats["flight_progress"]["sum"] += progress.sum()
            running_stats["flight_progress"]["sum_sq"] += (progress**2).sum()
            running_stats["flight_progress"]["count"] += count

        stats_for_segment = segment_df.select(
            [pl.sum(col).alias(f"{col}_sum") for col in STATE_FEATURES]
            + [(pl.col(col).pow(2)).sum().alias(f"{col}_sum_sq") for col in STATE_FEATURES]
        ).row(0, named=True)

        for feature in STATE_FEATURES:
            running_stats[feature]["sum"] += stats_for_segment[f"{feature}_sum"] or 0
            running_stats[feature]["sum_sq"] += stats_for_segment[f"{feature}_sum_sq"] or 0
            running_stats[feature]["count"] += count

    for feature, stats in running_stats.items():
        count = stats["count"]
        assert count > 2, f"not enough data for feature {feature}"
        mean = stats["sum"] / count
        variance = (stats["sum_sq"] / count) - (mean**2)
        assert variance >= 1e-9, f"variance for {feature} is negative or too small"
        std = np.sqrt(variance)

        standardisation_stats[feature] = {"mean": mean, "std": std}

    output_path = path_base / f"stats_{partition}.json"
    with open(output_path, "w") as f:
        json.dump(standardisation_stats, f, indent=2)
    logger.info(f"wrote stats to {output_path}")


#
# wrappers
#


def load_splits(
    partition: Partition, *, path_base: Path = PATH_PREPROCESSED
) -> dict[Split, list[SegmentId]]:
    with open(path_base / f"splits_{partition}.json") as f:
        return json.load(f)


def load_standardisation_stats(
    partition: Partition, *, path_base: Path = PATH_PREPROCESSED
) -> Stats:
    with open(path_base / f"stats_{partition}.json") as f:
        return json.load(f)


class IteratorData(NamedTuple):
    segments_df: pl.DataFrame
    traj_lf: pl.LazyFrame


def prepare_iterator_data(
    partition: Partition,
    segment_ids: Collection[SegmentId] | None = None,
    stats: Stats | None = None,
    path_base: Path = PATH_PREPROCESSED,
) -> IteratorData:
    """Prepares data required by the dataloader."""
    fuel_lf = raw.scan_fuel(partition)
    if segment_ids:
        fuel_lf = fuel_lf.filter(pl.col("idx").is_in(segment_ids))

    flight_list_lf = raw.scan_flight_list(partition)
    segments_df = (
        fuel_lf.join(
            flight_list_lf.select("flight_id", "takeoff", "landed", "aircraft_type"),
            on="flight_id",
        )
        .sort("flight_id")
        .collect()
    )

    # optimisation: select specific columns early to avoid OOM during large joins
    traj_cols = ["flight_id", "timestamp", *CORE_FEATURES]
    traj_lf = pl.scan_parquet(path_base / f"trajectories_{partition}.parquet").select(traj_cols)
    derived_path = path_base / f"derived_{partition}.parquet"
    if derived_path.exists() and WEATHER_FEATURES:
        derived_lf = pl.scan_parquet(derived_path)
        traj_lf = pl.concat(
            [traj_lf, derived_lf.select(WEATHER_FEATURES)], how="horizontal"
        )  # NOTE: we do not do a join here to avoid OOM: we already made sure rows align perfectly

    if stats is not None:
        standardisation_exprs = [
            ((pl.col(f) - stats[f]["mean"]) / stats[f]["std"]).alias(f) for f in STATE_FEATURES
        ]
        traj_lf = traj_lf.with_columns(standardisation_exprs)

    return IteratorData(segments_df=segments_df, traj_lf=traj_lf)


class TrajectoryInfo(TypedDict):
    idx: int
    flight_id: str
    start: datetime
    end: datetime
    fuel_kg: float
    takeoff: datetime
    landed: datetime
    aircraft_type: str


class Trajectory(NamedTuple):
    features_df: pl.DataFrame
    info: TrajectoryInfo


class TrajectoryIterator:
    """Yields the entire flight trajectory for each segment as polars DataFrames."""

    def __init__(
        self,
        partition: Partition,
        *,
        segment_ids: Collection[SegmentId] | None = None,
        shuffle_seed: int | None = None,
        stats: Stats | None = None,
        start_to_end_only: bool = False,
        path_base: Path = PATH_PREPROCESSED,
    ):
        """
        :param start_to_end_only: if False, yields the entire materialised flight trajectory.
            Note that collecting this iterator will use a lot of memory due to duplicates!
            Prefer using the torch iterator instead.
        """
        it_data = prepare_iterator_data(partition, segment_ids, stats, path_base)
        self.traj_lf = it_data.traj_lf
        self.segment_infos: list[TrajectoryInfo] = it_data.segments_df.to_dicts()  # type: ignore
        self.start_to_end_only = start_to_end_only
        self.stats = stats

        self.segments_by_flight: dict[FlightId, list[TrajectoryInfo]] = {}
        for info in self.segment_infos:
            self.segments_by_flight.setdefault(info["flight_id"], []).append(info)

        self.flight_ids_to_iterate = list(self.segments_by_flight.keys())
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(self.flight_ids_to_iterate)

    def __len__(self) -> int:
        return len(self.segment_infos)

    def __iter__(self) -> Iterator[Trajectory]:
        for flight_id in self.flight_ids_to_iterate:
            flight_traj_df = self.traj_lf.filter(pl.col("flight_id") == flight_id).collect()

            for segment_info in self.segments_by_flight[flight_id]:
                if self.start_to_end_only:
                    start_ts = segment_info["start"]
                    end_ts = segment_info["end"]

                    start_idx, end_idx = find_segment_indices(
                        flight_traj_df["timestamp"].to_numpy(),
                        np.datetime64(start_ts.isoformat()),
                        np.datetime64(end_ts.isoformat()),
                        xp=np,
                    )
                    segment_traj_df = flight_traj_df[start_idx:end_idx]
                    if segment_traj_df.height < 2:
                        logger.error(
                            f"skipping {flight_id}: found < 2 datapoints for segment "
                            f"({start_ts} - {end_ts}): {start_idx}..={end_idx}"
                        )
                        continue
                else:
                    segment_traj_df = flight_traj_df

                yield Trajectory(
                    features_df=segment_traj_df,
                    info=segment_info,
                )
