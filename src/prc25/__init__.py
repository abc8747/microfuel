from pathlib import Path
from typing import Annotated, Generic, Literal, NamedTuple, TypeAlias, TypeVar, get_args

import isqx
import isqx.usc

PATH_ROOT = Path(__file__).parent.parent.parent
PATH_DATA = PATH_ROOT / "data"
PATH_DATA_RAW = PATH_DATA / "raw"
PATH_PREPROCESSED = PATH_DATA / "preprocessed"
PATH_PLOTS_OUTPUT = PATH_DATA / "plots"
PATH_CHECKPOINTS = PATH_DATA / "checkpoints"
PATH_PREDICTIONS = PATH_DATA / "predictions"

FlightId: TypeAlias = str
"""Unique flight identifier: `prc_{}`"""
AircraftType: TypeAlias = Literal[
    "A20N",  # 29.17%
    "A320",  # 27.39%
    "A359",  # 14.23%
    "B788",  # 5.28%
    "B738",  # 4.99%
    "A332",  # 4.82%
    "A21N",  # 3.56%
    "A321",  # 2.58%
    "B789",  # 2.03%
    "B77W",  # 1.66%
    "A333",  # 1.41%
    "B772",  # 0.93%
    "B744",  # 0.82%
    "B737",  # 0.26%
    "B739",  # 0.25%
    "B38M",  # 0.18%
    "A319",  # 0.17%
    "A306",  # 0.09%
    "A388",  # 0.04%
    "B752",  # 0.03%
    "B748",  # 0.03%
    "B77L",  # 0.02%
    "B763",  # 0.02%
    "MD11",  # 0.01%
    "B39M",  # 0.01%
    "A318",  # 0.01%
]
AIRCRAFT_TYPES: tuple[AircraftType, ...] = get_args(AircraftType)
AirportIcao: TypeAlias = str
Partition: TypeAlias = Literal["phase1", "phase1_rank"]
Split: TypeAlias = Literal["train", "validation"]
SPLITS: tuple[Split, ...] = get_args(Split)

deg2rad = isqx.convert(isqx.RAD, isqx.DEG)
ft2m = isqx.convert(isqx.usc.FT, isqx.M)
knot2mps = isqx.convert(isqx.usc.KNOT, isqx.M_PERS)
fpm2mps = isqx.convert(isqx.usc.FT * isqx.MIN**-1, isqx.M_PERS)

_T = TypeVar("_T")


class Coordinate2D(NamedTuple, Generic[_T]):
    lng: Annotated[_T, isqx.LONGITUDE(isqx.DEG)]
    lat: Annotated[_T, isqx.LATITUDE(isqx.DEG)]
