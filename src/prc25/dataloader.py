"""Prepares variable-length batches to predict fuel burn from trajectory data.

We have:
1. Time series trajectory of a *full flight*.
2. *Segments* of fuel burn data (discrete intervals with start and end times).

Do NOT pad sequences: our sequences range from 2 to thousands of tokens.

Instead, we concatenate all sequences in the batch into one long tensor and
provide metadata to tell the kernel where each sequence begins and ends.

The Triton kernels in FLA are specifically designed to be varlen-aware.

See `chunk_bwd_kernel_dqkwg` kernel in `lit_gpt/gated_delta_rule_ops/fla_version/chunk_fla.py`.
"""

import logging
from collections import namedtuple
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from . import PATH_PREPROCESSED, Partition, Split
from .datasets import preprocessed, raw

if TYPE_CHECKING:
    from .datasets.preprocessed import SplitData

logger = logging.getLogger(__name__)


Sequence = namedtuple("Sequence", ["features", "target", "segment_id", "aircraft_type_idx"])
VarlenBatch = namedtuple("VarlenBatch", ["x", "y", "offsets", "segment_ids", "aircraft_type_idx"])


def get_split_flight_ids(partition: Partition, split: Split) -> list[str]:
    split_data: SplitData = preprocessed.train_validation_split(partition)
    return split_data[f"{split}_flight_ids"]  # type: ignore


def get_standardisation_stats(partition: Partition) -> tuple[np.ndarray, np.ndarray]:
    split_data: SplitData = preprocessed.train_validation_split(partition)
    stats = split_data["standardisation_stats"]
    mean_list: list[float] = []
    std_list: list[float] = []
    for feature in preprocessed.FEATURES:
        mean_list.append(stats[feature]["mean"])
        std_list.append(stats[feature]["std"])
    return np.array(mean_list, dtype=np.float32), np.array(std_list, dtype=np.float32)


def build_feature_dataframe(
    partition: Partition,
    flight_ids: list[str],
    min_seq_len: int,
    max_seq_len: int,
) -> pl.DataFrame:
    traj_lf = pl.scan_parquet(PATH_PREPROCESSED / f"trajectories_{partition}.parquet").filter(
        pl.col("flight_id").is_in(flight_ids)
    )
    fuel_lf = raw.scan_fuel(partition).filter(pl.col("flight_id").is_in(flight_ids))
    flight_list_lf = raw.scan_flight_list(partition)

    fuel_with_target_lf = fuel_lf.with_columns(
        ((pl.col("fuel_kg") / (pl.col("end") - pl.col("start")).dt.total_seconds()) + 1.0)
        .log()
        .alias("log_avg_fuel_burn_rate_kg_s_p1")
    )

    traj_with_actype_lf = traj_lf.join(
        flight_list_lf.select("flight_id", "aircraft_type"), on="flight_id"
    )
    valid_segments_lf = (
        traj_with_actype_lf.filter(pl.col("segment_id").is_not_null())
        .group_by(["flight_id", "segment_id"])
        .len()
        .filter(pl.col("len").is_between(min_seq_len, max_seq_len))
    )

    final_data_df = (
        traj_with_actype_lf.join(valid_segments_lf, on=["flight_id", "segment_id"])
        .join(
            fuel_with_target_lf,
            left_on=["flight_id", "segment_id"],
            right_on=["flight_id", "idx"],
        )
        .sort("flight_id", "segment_id", "time_since_takeoff")
        .select(
            "flight_id",
            "segment_id",
            "aircraft_type",
            *preprocessed.FEATURES,
            "log_avg_fuel_burn_rate_kg_s_p1",
        )
        .collect()
    )
    return final_data_df


def create_sequences_from_dataframe(
    df: pl.DataFrame, means: np.ndarray, stds: np.ndarray, ac_type_vocab: dict[str, int]
) -> list[Sequence]:
    if df.is_empty():
        return []
    df = df.with_columns(
        pl.col("aircraft_type")
        .map_elements(lambda x: ac_type_vocab.get(x, ac_type_vocab["UNK"]), return_dtype=pl.Int32)
        .alias("aircraft_type_idx")
    )
    return [
        Sequence(
            features=(
                group.select(preprocessed.FEATURES).to_numpy(order="c").astype(np.float32) - means
            )
            / stds,
            target=group.select("log_avg_fuel_burn_rate_kg_s_p1").item(0, 0),
            segment_id=key[1],
            aircraft_type_idx=group.select("aircraft_type_idx").item(0, 0),
        )
        for key, group in df.group_by(["flight_id", "segment_id"], maintain_order=True)
    ]


class VarlenDataset(Dataset):
    def __init__(
        self,
        partition: Partition,
        split: Split,
        min_seq_len: int = 2,
        max_seq_len: int = 65536,
    ):
        flight_ids = get_split_flight_ids(partition, split)
        self.means, self.stds = get_standardisation_stats(partition)

        train_flight_ids = get_split_flight_ids(partition, "train")
        flight_list_lf = raw.scan_flight_list(partition)
        ac_types = (
            flight_list_lf.filter(pl.col("flight_id").is_in(train_flight_ids))
            .select("aircraft_type")
            .unique()
            .sort("aircraft_type")
            .collect()["aircraft_type"]
            .to_list()
        )
        self.ac_type_vocab = {ac_type: i for i, ac_type in enumerate(ac_types)}
        self.ac_type_vocab["UNK"] = len(self.ac_type_vocab)

        final_data_df = build_feature_dataframe(partition, flight_ids, min_seq_len, max_seq_len)

        if final_data_df.is_empty():
            logger.warning(f"no valid data for {partition=}, {split=}")

        self.sequences = create_sequences_from_dataframe(
            final_data_df, self.means, self.stds, self.ac_type_vocab
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Sequence:
        return self.sequences[idx]


def collate_fn(batch_sequences: list[Sequence]) -> VarlenBatch:
    lengths = [len(seq.features) for seq in batch_sequences]

    x = torch.from_numpy(np.concatenate([seq.features for seq in batch_sequences], axis=0))
    y = torch.tensor([seq.target for seq in batch_sequences], dtype=torch.float32).unsqueeze(1)
    offsets = torch.from_numpy(np.cumsum([0, *lengths], dtype=np.int32))
    segment_ids = torch.tensor([seq.segment_id for seq in batch_sequences], dtype=torch.int32)
    aircraft_type_idx = torch.tensor(
        [seq.aircraft_type_idx for seq in batch_sequences], dtype=torch.long
    )

    return VarlenBatch(
        x=x, y=y, offsets=offsets, segment_ids=segment_ids, aircraft_type_idx=aircraft_type_idx
    )
