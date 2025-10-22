from pathlib import Path
from typing import Literal, TypeAlias, get_args

PATH_ROOT = Path(__file__).parent.parent.parent
PATH_DATA = PATH_ROOT / "data"
PATH_DATA_RAW = PATH_DATA / "raw"
PATH_PREPROCESSED = PATH_DATA / "preprocessed"
PATH_PLOTS_OUTPUT = PATH_DATA / "plots"
PATH_CHECKPOINTS = PATH_DATA / "checkpoints"
PATH_PREDICTIONS = PATH_DATA / "predictions"


Partition: TypeAlias = Literal["phase1", "rank"]
Split: TypeAlias = Literal["train", "validation"]
SPLITS: tuple[Split, ...] = get_args(Split)
