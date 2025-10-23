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

from __future__ import annotations

import logging
from collections import namedtuple

import numpy as np
import polars as pl
import torch
from rich.progress import track
from torch.utils.data import Dataset

from . import PATH_PREPROCESSED, Partition, Split
from .datasets import preprocessed, raw
from .datasets.preprocessed import TrajectoryIterator, load_standardisation_stats

logger = logging.getLogger(__name__)


Sequence = namedtuple(
    "Sequence", ["features", "target", "segment_id", "aircraft_type_idx", "duration_s"]
)
VarlenBatch = namedtuple(
    "VarlenBatch", ["x", "y", "offsets", "segment_ids", "aircraft_type_idx", "durations"]
)


class VarlenDataset(Dataset):
    def __init__(
        self,
        partition: Partition,
        split: Split,
        min_seq_len: int = 2,
        max_seq_len: int = 65536,
    ):
        traj_lf = pl.scan_parquet(PATH_PREPROCESSED / f"trajectories_{partition}_{split}.parquet")
        flight_ids = traj_lf.select("flight_id").unique().collect()["flight_id"].to_list()
        self.stats = load_standardisation_stats(partition)

        train_traj_lf = pl.scan_parquet(
            PATH_PREPROCESSED / f"trajectories_{partition}_train.parquet"
        )
        train_flight_ids = (
            train_traj_lf.select("flight_id").unique().collect()["flight_id"].to_list()
        )

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

        trajectory_iterator = TrajectoryIterator(
            partition=partition,
            split=split,
            flight_ids=flight_ids,
            stats=self.stats,
            start_to_end_only=True,
        )

        self.sequences: list[Sequence] = []
        for trajectory in track(
            trajectory_iterator,
            description=f"loading {split} data",
            total=len(trajectory_iterator),
        ):
            segment_features_df = trajectory.features_df
            if not (min_seq_len <= len(segment_features_df) <= max_seq_len):
                continue

            features_np = (
                segment_features_df.select(preprocessed.TRAJECTORY_FEATURES)
                .to_numpy(order="c")
                .astype(np.float32)
            )

            duration_s = (trajectory.info["end"] - trajectory.info["start"]).total_seconds()
            target = np.log((trajectory.info["fuel_kg"] / duration_s) + 1.0)

            ac_type_idx = self.ac_type_vocab.get(
                trajectory.info["aircraft_type"], self.ac_type_vocab["UNK"]
            )

            self.sequences.append(
                Sequence(
                    features=features_np,
                    target=target,
                    segment_id=trajectory.info["idx"],
                    aircraft_type_idx=ac_type_idx,
                    duration_s=duration_s,
                )
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
    durations = torch.tensor([seq.duration_s for seq in batch_sequences], dtype=torch.float32)

    return VarlenBatch(
        x=x,
        y=y,
        offsets=offsets,
        segment_ids=segment_ids,
        aircraft_type_idx=aircraft_type_idx,
        durations=durations,
    )
