from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

import polars as pl

from .. import PATH_DATA, PATH_DATA_RAW


class Config(TypedDict):
    team_id: int
    team_name: str
    bucket_access_key: str
    bucket_access_secret: str


# NOTE: end dataset wrappers
# the following functions generates the dataset


def load_config(fp: Path = PATH_DATA / "config.toml") -> Config:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    with open(fp, "r") as f:
        return tomllib.loads(f.read())  # type: ignore


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
) -> None:
    """Download data from S3 using MinIO client.
    Not using boto3 because it is extremely slow."""
    path_out.mkdir(parents=True, exist_ok=True)
    
    setup_mc_alias(bucket_access_key, bucket_access_secret, endpoint_url, alias_name)
    
    cmd = f"mc cp --recursive {alias_name}/{bucket_name}/ {path_out}/"
    os.system(cmd)
