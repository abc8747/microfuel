from __future__ import annotations

import logging
from collections import Counter, namedtuple

import numpy as np
import polars as pl
import torch
from rich.progress import track
from torch.utils.data import Dataset

from . import AIRCRAFT_TYPES, Partition, Split
from .datasets import preprocessed, raw

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


def _compute_and_standardise_features(
    all_trajs_df: pl.DataFrame,
    segments_with_boundaries_df: pl.DataFrame,
    stats: preprocessed.Stats,
) -> pl.DataFrame:
    """Computes and standardizes features for all trajectories."""
    flight_info_df = (
        segments_with_boundaries_df.select("flight_id", "takeoff", "landed")
        .unique("flight_id")
        .with_columns(
            flight_duration_s=(
                (pl.col("landed") - pl.col("takeoff")).dt.total_seconds(fractional=True)
            )
        )
    )
    df = all_trajs_df.join(flight_info_df, on="flight_id", how="left")

    standardisation_exprs = [
        ((pl.col(f) - stats[f]["mean"]) / stats[f]["std"]).alias(f)
        for f in preprocessed.STATE_FEATURES
    ]
    # NOTE: feature names in the final tensor must match MODEL_INPUT_FEATURES order
    return df.with_columns(
        flight_progress=(
            (pl.col("timestamp") - pl.col("takeoff")).dt.total_seconds(fractional=True)
            / pl.col("flight_duration_s")
        )
    ).with_columns(
        flight_progress=(pl.col("flight_progress") - stats["flight_progress"]["mean"])
        / stats["flight_progress"]["std"],
        flight_duration=(pl.col("flight_duration_s") - stats["flight_duration"]["mean"])
        / stats["flight_duration"]["std"],
        *standardisation_exprs,
    )


def _prepare_tensors(
    partition: Partition,
    flight_ids: list[str],
    segment_ids: list[int] | None,
    stats: preprocessed.Stats,
    ac_type_vocab: dict[str, int],
) -> tuple[torch.Tensor, list[SequenceInfo]]:
    """Loads data, computes features, and prepares tensors for the Dataset."""
    it_data = preprocessed.prepare_iterator_data(partition, segment_ids, stats=None)
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
    segments_with_boundaries_df = it_data.segments_df.join(
        flight_id_boundaries, on="flight_id", how="inner"
    )

    featured_df = _compute_and_standardise_features(
        all_trajs_df, segments_with_boundaries_df, stats
    )

    all_features_tensor = featured_df.select(preprocessed.MODEL_INPUT_FEATURES).to_torch(
        dtype=pl.Float32
    )
    timestamps_all_np = featured_df["timestamp"].to_numpy()

    sequences: list[SequenceInfo] = []
    for row in track(
        segments_with_boundaries_df.iter_rows(named=True),
        description=f"indexing sequences for {partition}/?",
        total=len(segments_with_boundaries_df),
    ):
        flight_start_idx, flight_len = row["start_idx"], row["length"]
        flight_timestamps = timestamps_all_np[flight_start_idx : flight_start_idx + flight_len]

        start_offset, end_offset = preprocessed.find_segment_indices(
            flight_timestamps,
            np.datetime64(row["start"].isoformat()),
            np.datetime64(row["end"].isoformat()),
            xp=np,
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
        ac_type_idx = ac_type_vocab[row["aircraft_type"]]

        sequences.append(
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
    return all_features_tensor, sequences


class VarlenDataset(Dataset):
    def __init__(self, partition: Partition, split: Split | None):
        if split:
            splits = preprocessed.load_splits(partition)
            segment_ids = splits[split]
            flight_ids = (
                raw.scan_fuel(partition)
                .filter(pl.col("idx").is_in(segment_ids))
                .select("flight_id")
                .unique()
                .collect()["flight_id"]
                .to_list()
            )
        else:
            segment_ids = None
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

        self.all_features, self.sequences = _prepare_tensors(
            partition, flight_ids, segment_ids, self.stats, self.ac_type_vocab
        )

        counts = Counter(s.aircraft_type_idx for s in self.sequences)
        self.class_counts = torch.tensor([counts[i] for i in range(len(self.ac_type_vocab))])

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
