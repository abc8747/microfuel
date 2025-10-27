from pathlib import Path
from typing import Literal, TypeAlias, get_args

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
TypeCode: TypeAlias = str  # TODO: concretise this to get stable name -> int enum mappings in torch
"""Aircraft type."""
Partition: TypeAlias = Literal["phase1", "rank"]
Split: TypeAlias = Literal["train", "validation"]
SPLITS: tuple[Split, ...] = get_args(Split)

deg2rad = isqx.convert(isqx.RAD, isqx.DEG)
ft2m = isqx.convert(isqx.usc.FT, isqx.M)
knot2mps = isqx.convert(isqx.usc.KNOT, isqx.M_PERS)
fpm2mps = isqx.convert(isqx.usc.FT * isqx.MIN**-1, isqx.M_PERS)
