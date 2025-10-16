from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal, TypedDict

import polars as pl

from .. import PATH_DATA, PATH_DATA_RAW


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
#

Partition = Literal["train", "rank"] | str


def scan_fuel(partition: Partition = "train") -> pl.LazyFrame:
    filename = (
        "fuel_rank_submission.parquet" if partition == "rank" else f"fuel_{partition}.parquet"
    )
    fp = PATH_DATA_RAW / filename
    return pl.scan_parquet(fp)


def scan_flight_list(partition: Partition = "train") -> pl.LazyFrame:
    filename = f"flight_list_{partition}.parquet"
    fp = PATH_DATA_RAW / filename
    return pl.scan_parquet(fp)


def scan_airports() -> pl.LazyFrame:
    fp = PATH_DATA_RAW / "apt.parquet"
    return pl.scan_parquet(fp)


def scan_all_trajectories(partition: Partition = "train") -> pl.LazyFrame:
    path = PATH_DATA_RAW / f"flights_{partition}"
    return pl.scan_parquet(f"{path}/*.parquet")


def fp_trajectory(flight_id: str, partition: str = "train") -> Path:
    return PATH_DATA_RAW / f"flights_{partition}" / f"{flight_id}.parquet"


def scan_trajectory(flight_id: str, partition: str = "train") -> pl.LazyFrame | None:
    fp = fp_trajectory(flight_id, partition)
    if not fp.exists():
        return None
    return pl.scan_parquet(fp)
