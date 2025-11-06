from __future__ import annotations

import logging
from collections import namedtuple

import numpy as np
import polars as pl
import torch
from rich.progress import track
from torch.utils.data import Dataset

from . import AIRCRAFT_TYPES, Partition, Split
from .datasets import preprocessed

logger = logging.getLogger(__name__)


SequenceInfo = namedtuple(
    "SequenceInfo",
    [
        "flight_indices",
        "segment_indices_relative",
        "target",
        "segment_id",
        "aircraft_type_idx",
        "duration_s",
        "flight_id",
    ],
)
Sequence = namedtuple(
    "Sequence",
    ["features", "target", "segment_id", "aircraft_type_idx", "duration_s", "flight_id"],
)
VarlenBatch = namedtuple(
    "VarlenBatch",
    [
        "x",
        "y",
        "cu_seqlens",
        "segment_ids",
        "aircraft_type_idx",
        "durations",
    ],
)


class VarlenDataset(Dataset):
    def __init__(self, partition: Partition, split: Split | None):
        if split:
            splits = preprocessed.load_splits(partition)
            flight_ids = splits[split]
        else:
            from .datasets import raw

            flight_ids = (
                raw.scan_fuel(partition)
                .select("flight_id")
                .unique()
                .collect()["flight_id"]
                .to_list()
            )

        # always get train stats for submission
        stats_partition: Partition = partition.removesuffix("_rank")  # type: ignore
        self.stats = preprocessed.load_standardisation_stats(stats_partition)

        self.ac_type_vocab = {ac_type: i for i, ac_type in enumerate(AIRCRAFT_TYPES)}
        self.ac_type_vocab["UNK"] = len(self.ac_type_vocab)
        # TODO: for some reason removing UNK worsens performance significantly
        # but UNK is unused so maybe something with the initialisation.

        it_data = preprocessed.prepare_iterator_data(partition, flight_ids, self.stats)
        all_trajs_df = (
            it_data.traj_lf.filter(pl.col("flight_id").is_in(flight_ids))
            .sort("flight_id", "timestamp")
            .collect()
        )

        flight_id_boundaries = (
            all_trajs_df.with_row_index()
            .group_by("flight_id", maintain_order=True)
            .agg(pl.first("index").alias("start_idx"), pl.len().alias("length"))
        )
        segments_with_boundaries = it_data.segments_df.join(
            flight_id_boundaries, on="flight_id", how="inner"
        )

        self.all_features = all_trajs_df.select(preprocessed.TRAJECTORY_FEATURES).to_torch(
            dtype=pl.Float32
        )
        # we no longer have `timestamp` here so we must locate a segment using `time_since_takeoff`
        time_since_takeoff_all_std = self.all_features[:, 0]
        tst_stats = self.stats["time_since_takeoff"]
        tst_mean, tst_std = tst_stats["mean"], tst_stats["std"]

        self.sequences: list[SequenceInfo] = []
        for row in track(
            segments_with_boundaries.iter_rows(named=True),
            description=f"loading {split or partition} data",
            total=len(segments_with_boundaries),
        ):
            flight_start_idx, flight_len = row["start_idx"], row["length"]
            time_since_takeoff_flight_std = time_since_takeoff_all_std[
                flight_start_idx : flight_start_idx + flight_len
            ].contiguous()

            start_relative = (row["start"] - row["takeoff"]).total_seconds()
            end_relative = (row["end"] - row["takeoff"]).total_seconds()

            start_relative_std = (start_relative - tst_mean) / tst_std
            end_relative_std = (end_relative - tst_mean) / tst_std

            start_offset, end_offset = preprocessed.find_segment_indices(
                time_since_takeoff_flight_std,
                start_relative_std - 1e-9,
                end_relative_std + 1e-9,
                xp=torch,
            )

            if (end_offset - start_offset) < 2:
                logger.error(
                    f"expected {row['flight_id']}/{row['idx']} to have "
                    f"at least two datapoints, but got {end_offset - start_offset} points for "
                    f"({row['start']} - {row['end']})."
                )
                continue

            flight_end_idx = flight_start_idx + flight_len
            segment_indices_relative = (start_offset.item(), end_offset.item())

            duration_s = (row["end"] - row["start"]).total_seconds()
            target = np.log1p((row["fuel_kg"] or np.nan) / duration_s)
            ac_type_idx = self.ac_type_vocab.get(row["aircraft_type"], self.ac_type_vocab["UNK"])

            self.sequences.append(
                SequenceInfo(
                    flight_indices=(flight_start_idx, flight_end_idx),
                    segment_indices_relative=segment_indices_relative,
                    target=target,
                    segment_id=row["idx"],
                    aircraft_type_idx=ac_type_idx,
                    duration_s=duration_s,
                    flight_id=row["flight_id"],
                )
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Sequence:
        seq_info = self.sequences[idx]
        flight_start_abs, _ = seq_info.flight_indices
        segment_start_rel, segment_end_rel = seq_info.segment_indices_relative
        segment_start_abs = flight_start_abs + segment_start_rel
        segment_end_abs = flight_start_abs + segment_end_rel
        features = self.all_features[segment_start_abs:segment_end_abs]
        return Sequence(
            features=features,
            target=seq_info.target,
            segment_id=seq_info.segment_id,
            aircraft_type_idx=seq_info.aircraft_type_idx,
            duration_s=seq_info.duration_s,
            flight_id=seq_info.flight_id,
        )


def collate_fn(batch_sequences: list[Sequence]) -> VarlenBatch:
    lengths = [len(seq.features) for seq in batch_sequences]

    x = torch.cat([seq.features for seq in batch_sequences], dim=0)
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
