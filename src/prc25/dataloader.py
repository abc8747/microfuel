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
    "VarlenBatch", ["x", "y", "cu_seqlens", "segment_ids", "aircraft_type_idx", "durations"]
)


class VarlenDataset(Dataset):
    def __init__(self, partition: Partition, split: Split):
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
            # NOTE: some segments can have no data points...
            if (height := trajectory.features_df.height) < 2:
                logger.warning(
                    f"skipping trajectory {trajectory.info['idx']} "
                    f"({trajectory.info['start']}-{trajectory.info['end']}) with {height=}."
                )
                continue

            features_np = (
                trajectory.features_df.select(preprocessed.TRAJECTORY_FEATURES)
                .to_numpy(order="c")
                .astype(np.float32)
            )

            duration_s = (trajectory.info["end"] - trajectory.info["start"]).total_seconds()
            target = np.log((trajectory.info["fuel_kg"] / duration_s) + 1.0)

            ac_type_idx = self.ac_type_vocab.get(
                trajectory.info["aircraft_type"], self.ac_type_vocab["UNK"]
            )

            # collect here to amortise cost, but this is unscalable for full flight datasets.
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
    cu_seqlens = torch.from_numpy(np.cumsum([0, *lengths], dtype=np.int32))
    segment_ids = torch.tensor([seq.segment_id for seq in batch_sequences], dtype=torch.int32)
    aircraft_type_idx = torch.tensor(
        [seq.aircraft_type_idx for seq in batch_sequences], dtype=torch.long
    )
    durations = torch.tensor([seq.duration_s for seq in batch_sequences], dtype=torch.float32)

    return VarlenBatch(
        x=x,
        y=y,
        cu_seqlens=cu_seqlens,
        segment_ids=segment_ids,
        aircraft_type_idx=aircraft_type_idx,
        durations=durations,
    )
