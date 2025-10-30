from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

import polars as pl

from .. import PATH_DATA, PATH_DATA_RAW, AircraftType, FlightId, Partition

if TYPE_CHECKING:
    import isqx
    import isqx.aerospace
    import isqx.usc


class Config(TypedDict):
    team_id: int
    team_name: str
    bucket_access_key: str
    bucket_access_secret: str


def load_config(fp: Path = PATH_DATA / "config.toml") -> Config:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    with open(fp, "rb") as f:
        return tomllib.load(f)  # type: ignore


def setup_mc_alias(
    bucket_access_key: str,
    bucket_access_secret: str,
    endpoint_url: str = "https://s3.opensky-network.org:443",
    alias_name: str = "prc25",
) -> int:
    cmd = f"mc alias set {alias_name} {endpoint_url} {bucket_access_key} {bucket_access_secret}"
    return os.system(cmd)


def download_from_s3(
    bucket_access_key: str,
    bucket_access_secret: str,
    *,
    path_out: Path = PATH_DATA_RAW,
    bucket_name: str = "prc-2025-datasets",
    endpoint_url: str = "https://s3.opensky-network.org:443",
    alias_name: str = "prc2025",
) -> int:
    """Download data from S3 using MinIO client.
    Not using boto3 because it is extremely slow."""
    path_out.mkdir(parents=True, exist_ok=True)
    setup_mc_alias(bucket_access_key, bucket_access_secret, endpoint_url, alias_name)
    cmd = f"mc cp --recursive {alias_name}/{bucket_name}/ {path_out}/"
    return os.system(cmd)


#
# wrappers
# makes sure timestamps in parquet are always read as UTC
# because stdlib's `datetime` often treat naiive timestamps as local time
#


class FuelRecord(TypedDict):
    idx: int
    flight_id: FlightId
    start: datetime
    end: datetime
    fuel_kg: Annotated[float, isqx.MASS(isqx.KG)]


def scan_fuel(
    partition: Partition = "phase1", *, path_base: Path = PATH_DATA_RAW
) -> Annotated[pl.LazyFrame, FuelRecord]:
    fp = path_base / f"fuel_{partition}.parquet"
    # timestamps are in nanoseconds
    return pl.scan_parquet(fp).with_columns(
        pl.col("start").dt.replace_time_zone("UTC"), pl.col("end").dt.replace_time_zone("UTC")
    )


class FlightListRecord(TypedDict):
    flight_id: FlightId
    flight_date: datetime
    takeoff: datetime
    landed: datetime
    origin_icao: str
    origin_name: str
    destination_icao: str
    destination_name: str
    aircraft_type: AircraftType


def scan_flight_list(
    partition: Partition = "phase1", *, path_base: Path = PATH_DATA_RAW
) -> Annotated[pl.LazyFrame, FlightListRecord]:
    fp = path_base / f"flight_list_{partition}.parquet"
    # timestamp are in seconds
    return pl.scan_parquet(fp).with_columns(
        pl.col("takeoff").dt.replace_time_zone("UTC"), pl.col("landed").dt.replace_time_zone("UTC")
    )


class AirportRecord(TypedDict):
    icao: str
    latitude: Annotated[float, isqx.LATITUDE(isqx.DEG)]
    longitude: Annotated[float, isqx.LONGITUDE(isqx.DEG)]
    elevation: Annotated[float, isqx.aerospace.GEOMETRIC_ALTITUDE(isqx.usc.FT)] | None


def scan_airports(*, path_base: Path = PATH_DATA_RAW) -> Annotated[pl.LazyFrame, AirportRecord]:
    fp = path_base / "apt.parquet"
    return pl.scan_parquet(fp)


class TrajectoryRecord(TypedDict):
    timestamp: datetime
    flight_id: FlightId
    typecode: AircraftType
    latitude: Annotated[float, isqx.LATITUDE(isqx.DEG)] | None
    """Latitude, encoded via Compact Positional Reporting (CPR, tc=9-18, 20-22)
    We do not have access to uncertainty/quantisation, can be anywhere from:

    - navigational integrity category: nic=11 (rc < 7.5m)..nic=8 (rc < 185m)
    - navigational accuracy category: nacp=10 (epu < 10m)..nacp=8 (epu < 93m)"""
    longitude: Annotated[float, isqx.LONGITUDE(isqx.DEG)] | None  # see above.
    altitude: Annotated[float, isqx.aerospace.PRESSURE_ALTITUDE(isqx.usc.FT)] | None
    """Barometric altitude (tc=9-18, 12-bit field). Not to be confused with GNSS altitude (tc=20-22)

    Quantisation: 'q' bit (bit 8 of the field):
    - q=1: 25-foot increments. altitude = (decimal value of 11 bits) * 25 - 1000 ft.
    - q=0: 100-foot increments, using gray code for altitudes > 50,175 ft.

    Uncertainty: depends on barometric altitude quality (baq)."""
    groundspeed: Annotated[float, isqx.aerospace.GS(isqx.usc.KNOT)] | None
    """Ground speed (GNSS or inertial reference system, tc=19, subtypes1-2).

    Not transmitted directly, encoded as two signed velocity components (east-west velocity,
    north-south velocity):

    - groundspeed = sqrt(vew^2 + vns^2)
    - track angle = atan2(vew, vns)

    Quantisation: 1 kt (subsonic), 4 kt (supersonic).
    Uncertainty: nacv=4 (< 0.3m/s), nacv=3 (< 1.0m/s), nacv=2 (< 3.0m/s), nacv=1 (< 10.0m/s)"""
    track: Annotated[float, isqx.DEG] | None  # see above.
    vertical_rate: Annotated[float, isqx.aerospace.VS(isqx.usc.FT * isqx.MIN**-1)] | None
    """Vertical rate (`vrsrc` specifies origin: GNSS or barometric, tc=19).

    a sign bit indicates climb or descent. a 9-bit value (vr) encodes the magnitude.
    vertical speed (ft/min) = 64 * (vr - 1).

    Uncertainty: linked to vertical component of nacv."""
    mach: Annotated[float, isqx.MACH_NUMBER] | None
    """Mach number (Mode S, BDS 6,0, 10 bits, mb 25-34).

    Quantisation: 0.004."""
    TAS: Annotated[float, isqx.aerospace.TAS(isqx.usc.KNOT)] | None
    """True airspeed.

    - ADS-B (tc=19, subtype 3/4) - Quantisation: 1 kt (subtype 3), 4 kt (subtype 4).
    - Mode S (BDS 5,0 track and turn report, 10 bits, mb 47-56) - Quantisation: 2 kt"""
    CAS: Annotated[float, isqx.aerospace.CAS(isqx.usc.KNOT)] | None
    """Calibrated airspeed. Not broadcast, but likely derived from indicated airspeed (BDS 6,0).

    Quantisation: 1 kt."""
    source: Literal["adsb", "acars", "artificial"]


def scan_all_trajectories(
    partition: Partition = "phase1", *, path_base: Path = PATH_DATA_RAW
) -> Annotated[pl.LazyFrame, TrajectoryRecord]:
    fp = path_base / f"flights_{partition}"
    return pl.scan_parquet(f"{fp}/*.parquet").with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC"),
    )


def scan_trajectory(
    flight_id: str, partition: Partition = "phase1", *, path_base: Path = PATH_DATA_RAW
) -> Annotated[pl.LazyFrame, TrajectoryRecord]:
    fp = path_base / f"flights_{partition}" / f"{flight_id}.parquet"
    return pl.scan_parquet(fp).with_columns(
        pl.col("timestamp").dt.replace_time_zone("UTC"),
    )
