from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer
from matplotlib.ticker import ScalarFormatter
from rich.logging import RichHandler
from rich.progress import track

import microfuel.plot as p
from microfuel import AIRCRAFT_TYPES, PATH_PLOTS_OUTPUT, Partition, Split
from microfuel.datasets import raw

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from microfuel.dataloader import VarlenBatch

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command()
def flight_duration_cdf() -> None:
    fig = p.default_fig(figsize=(10, 8))
    ax = fig.add_subplot(111)
    df = (
        raw.scan_flight_list("phase1")
        .with_columns(
            duration_hours=(pl.col("landed") - pl.col("takeoff")).dt.total_seconds() / 3600
        )
        .collect()
    )
    total = len(df)

    top_ac_types = (
        df.group_by("aircraft_type")
        .len()
        .with_columns(pl.col("aircraft_type").cast(pl.Enum(AIRCRAFT_TYPES)))
        .sort("aircraft_type")
    )
    num_types = len(top_ac_types)

    for i, row in enumerate(top_ac_types.iter_rows(named=True)):
        ac_type = row["aircraft_type"]
        count = row["len"]
        fraction = count / total
        durations = np.sort(
            df.filter(pl.col("aircraft_type") == ac_type)["duration_hours"].to_numpy()
        )
        cdf = np.arange(1, len(durations) + 1) / len(durations) * 100

        color = f"C{i % 10}"
        ax.plot(durations, cdf, color=color, linewidth=10 * max(fraction, 0.1))

        y_position = (i / (num_types - 1)) * 100 if num_types > 1 else 50
        cdf_idx = np.searchsorted(cdf, y_position)
        duration_at_y = durations[cdf_idx] if cdf_idx < len(durations) else durations[-1]
        ax.text(
            duration_at_y,
            y_position,
            f" {ac_type} ({fraction:.1%})",
            va="center",
            ha="left",
            fontsize=36 * max(fraction, 0.2),
            color=color,
        )

    ax.set_xlabel("Flight Duration (hours)")
    ax.set_ylabel("Cumulative Frequency (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    output_path = PATH_PLOTS_OUTPUT / "flight_duration.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {output_path}")


@app.command()
def speed_alt_fuel_burn(max_points_per_actype: float | None = 1000) -> None:
    """This is a very slow plot: takes ~1 minute"""
    fig = p.default_fig(figsize=(20, 20))

    flight_list_lf = raw.scan_flight_list("phase1")

    top_ac_types = list(AIRCRAFT_TYPES[:16])

    if max_points_per_actype is not None:
        sampled_dfs: list[pl.DataFrame] = []
        for ac_type in top_ac_types:
            ac_sample_df = flight_list_lf.filter(pl.col("aircraft_type") == ac_type).collect()
            sampled_dfs.append(
                ac_sample_df.sample(
                    n=min(ac_sample_df.height, int(max_points_per_actype)), shuffle=True, seed=13
                )
            )
        flight_list_sample = pl.concat(sampled_dfs)
    else:
        flight_list_sample = flight_list_lf.collect()

    fuel_df = (
        raw.scan_fuel("phase1")
        .with_columns(
            (pl.col("fuel_kg") / (pl.col("end") - pl.col("start")).dt.total_seconds()).alias(
                "avg_fuel_burn_rate_kg_s"
            )
        )
        .collect()
    )

    plot_df = flight_list_sample.join(fuel_df, on="flight_id")
    cmap = plt.get_cmap("turbo")

    axes = fig.subplots(4, 4, sharex=True, sharey=True)

    for i, ac_type in enumerate(track(top_ac_types, description="Processing aircraft types")):
        ax: Axes = axes.flatten()[i]
        data_subset = plot_df.filter(pl.col("aircraft_type") == ac_type)

        p_lower = max(0, data_subset["avg_fuel_burn_rate_kg_s"].quantile(0.01) or 0)
        p_upper = min(data_subset["avg_fuel_burn_rate_kg_s"].quantile(0.95) or 10, 10)

        for row in track(
            data_subset.iter_rows(named=True),
            total=data_subset.height,
            description=ac_type,
        ):
            traj_lf = raw.scan_trajectory(row["flight_id"], "phase1")
            traj_df = (
                traj_lf.filter(
                    (pl.col("timestamp") < row["end"])
                    & (pl.col("timestamp") > row["start"])
                    & (pl.col("groundspeed") < 600)
                    & (pl.col("altitude") < 50000)
                )
                .select("groundspeed", "altitude")
                .collect()
            )

            fuel_rate = row["avg_fuel_burn_rate_kg_s"]
            if fuel_rate < p_lower or fuel_rate > p_upper:
                continue

            norm_fuel = (fuel_rate - p_lower) / (p_upper - p_lower) if p_upper > p_lower else 0.5
            color = cmap(norm_fuel)

            ax.plot(
                *traj_df.select("groundspeed", "altitude"),
                color=color,
                alpha=0.1,
                linewidth=0.5,
            )

        sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(vmin=p_lower, vmax=p_upper))
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical")
        cbar.set_label("Avg Fuel Burn (kg/s)", fontsize=8)
        cbar.ax.tick_params(labelsize=6)

        ax.set_title(f"{ac_type}")
        ax.grid(True, alpha=0.3)

        if i % 4 == 0:
            ax.set_ylabel("Altitude (ft)")
        if i // 4 == 3:
            ax.set_xlabel("Groundspeed (knots)")

    output_path = PATH_PLOTS_OUTPUT / "speed_alt_fuel_burn.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(f"wrote {output_path}")


LBS_TO_KG = 0.45359237


@app.command()
def fuel_quantisation(tolerance: float = 0.01) -> None:
    fuel_lf = raw.scan_fuel("phase1")

    df = (
        fuel_lf.filter(pl.col("fuel_kg") > 0)
        .with_columns(
            pl.when(
                ((pl.col("fuel_kg") / LBS_TO_KG) - (pl.col("fuel_kg") / LBS_TO_KG).round()).abs()
                < tolerance
            )
            .then(pl.lit("imperial"))
            .when((pl.col("fuel_kg") - pl.col("fuel_kg").round()).abs() < tolerance)
            .then(pl.lit("metric"))
            .otherwise(pl.lit("neither"))
            .alias("classification")
        )
        .collect()
    )

    logger.info(df.group_by("classification").len().sort("classification"))

    ranges = [(0, 100), (100, 1000), (1000, 10000), (10000, 100000)]
    axes: list[Axes]
    fig, axes = plt.subplots(len(ranges), 1, figsize=(16, 12), sharex=False, sharey=True)

    colors = {"metric": "C1", "imperial": "C2", "neither": "C3"}
    y_positions = {"imperial": 1, "metric": 2, "neither": 3}

    for ax, (min_val, max_val) in zip(axes, ranges):
        range_df = df.filter(pl.col("fuel_kg").is_between(min_val, max_val, closed="left"))

        for classification in ["imperial", "metric", "neither"]:
            data = range_df.filter(pl.col("classification") == classification)["fuel_kg"].to_numpy()
            ax.scatter(
                data,
                np.full_like(data, y_positions[classification]),
                color=colors[classification],
                s=1,
                alpha=0.05,
                edgecolor="none",
            )

        ax.set_title(f"Range: {min_val} kg to {max_val} kg", loc="left")
        ax.set_xlabel("Fuel Burn (kg)")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(min_val, max_val)

    axes[0].set_yticks(list(y_positions.values()))
    axes[0].set_yticklabels(["imperial (lbs)", "metric (kg)", "neither"])
    for ax in axes:
        ax.tick_params(axis="y", length=0)

    output_path = PATH_PLOTS_OUTPUT / "fuel_quantisation.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"wrote {output_path}")


def _calculate_time_gaps(lf: pl.LazyFrame) -> pl.Series:
    return (
        lf.sort("flight_id", "timestamp")
        .with_columns(
            (pl.col("timestamp").diff().over("flight_id").dt.total_seconds(fractional=True)).alias(
                "gap_s"
            )
        )
        .select("gap_s")
        .drop_nulls()
        .filter(pl.col("gap_s") > 0)
        .collect()["gap_s"]
    )


def _plot_cdf(ax: Axes, data: np.ndarray, label: str, color: str) -> None:
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    ax.plot(sorted_data, cdf, linestyle="-", label=label, color=color)


@app.command()
def time_gap_cdf() -> None:
    traj_lf = raw.scan_all_trajectories("phase1")
    all_gaps = _calculate_time_gaps(traj_lf)
    acars_gaps = _calculate_time_gaps(traj_lf.filter(pl.col("source") == "acars"))
    fuel_lf = raw.scan_fuel("phase1")
    segment_lengths = (
        fuel_lf.with_columns(
            (pl.col("end") - pl.col("start")).dt.total_seconds(fractional=True).alias("length_s")
        )
        .filter(pl.col("length_s") > 0)
        .collect()["length_s"]
    )

    p._init_style()
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    _plot_cdf(ax, all_gaps.to_numpy(), "time gaps (all)", "C0")
    _plot_cdf(ax, acars_gaps.to_numpy(), "time gaps (acars)", "C2")
    _plot_cdf(ax, segment_lengths.to_numpy(), "segment lengths (fuel data)", "C3")

    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Cumulative Frequency (%)")
    ax.grid(True, which="both", linewidth=0.5)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()

    output_path = PATH_PLOTS_OUTPUT / "distributions.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote plot to {output_path}")


@app.command()
def fuel_burn_segment_duration(partition: Partition = "phase1"):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    from rich.progress import track

    from microfuel import AIRCRAFT_TYPES
    from microfuel.datasets import raw

    fuel_df = raw.scan_fuel(partition).collect()
    flight_list_df = raw.scan_flight_list(partition).select("flight_id", "aircraft_type").collect()
    df = fuel_df.join(flight_list_df, on="flight_id").with_columns(
        (pl.col("end") - pl.col("start")).dt.total_seconds().alias("duration_s")
    )

    N_ROWS = 5
    top_ac_types = AIRCRAFT_TYPES[: N_ROWS * N_ROWS]
    fig = plt.figure(figsize=(5 * N_ROWS, 5 * N_ROWS))
    gs = gridspec.GridSpec(N_ROWS, N_ROWS, hspace=0.05 * N_ROWS, wspace=0.05 * N_ROWS)
    axes = [fig.add_subplot(gs[i]) for i in range(N_ROWS * N_ROWS)]

    for i, ac_type in enumerate(track(top_ac_types)):
        ax = axes[i]
        subset = df.filter(pl.col("aircraft_type") == ac_type)
        duration = subset["duration_s"].to_numpy()
        fuel = subset["fuel_kg"].to_numpy()

        color = f"C{i % 10}"

        ax.scatter(
            duration,
            fuel,
            s=0.3,
            linewidth=0,
            color=color,
        )

        title = f"{ac_type} (n={len(subset)})"
        valid = (duration > 0) & (fuel > 0)
        if valid.sum() > 1:
            log_duration = np.log10(duration[valid])
            log_fuel = np.log10(fuel[valid])
            coeffs = np.polyfit(log_duration, log_fuel, 1)
            slope, intercept = coeffs

            duration_sorted = np.sort(duration[valid])
            trendline = 10 ** (slope * np.log10(duration_sorted) + intercept)

            ax.plot(duration_sorted, trendline, color=color, linewidth=0.5, alpha=0.8)
            title += f"\n$y = {10**intercept:.2f} x^{{{slope:.2f}}}$"

        ax.set_title(title, fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2, which="both")

        if i // N_ROWS == 4:
            ax.set_xlabel("Segment Duration (s)")
        if i % N_ROWS == 0:
            ax.set_ylabel("Fuel Burn (kg)")

    output_path = PATH_PLOTS_OUTPUT / "fuel_burn_segment_duration.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote plot to {output_path}")


#
# preprocessed
#


def _load_segment_info(partition: Partition) -> pl.LazyFrame:
    from microfuel import PATH_PREPROCESSED

    fname = f"segment_info_{partition}.parquet"
    path = PATH_PREPROCESSED / fname
    if not path.exists():
        raise FileNotFoundError(
            f"expected segments stats file to be found at {path}.\nplease run `uv run "
            f"scripts/main.py create-segment-info --partition {partition}` first."
        )
    return pl.scan_parquet(path)


@app.command()
def segment_distributions() -> None:
    import matplotlib.gridspec as gridspec
    import numpy as np

    df_rank = _load_segment_info("phase1_rank").collect()
    seq_lens_rank = df_rank["seq_len"].to_numpy()
    durations_rank = df_rank["duration_s"].to_numpy()

    df_all = _load_segment_info("phase1").collect()
    seq_lens_all = df_all["seq_len"].to_numpy()
    durations_all = df_all["duration_s"].to_numpy()

    fig = plt.figure(figsize=(16, 32))
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 2, 2], hspace=0.5)

    ax_seq_len_cdf = fig.add_subplot(gs[0])
    ax_duration_cdf = fig.add_subplot(gs[1])
    ax_ac_dist = fig.add_subplot(gs[2])

    _plot_cdf(ax_seq_len_cdf, seq_lens_rank, "phase1_rank", "C0")
    _plot_cdf(ax_seq_len_cdf, seq_lens_all, "phase1", "C1")
    ax_seq_len_cdf.set_xscale("log")
    ax_seq_len_cdf.set_ylim(0, 100)
    ax_seq_len_cdf.set_xlabel("Segment Sequence Length")
    ax_seq_len_cdf.set_ylabel("Cumulative Frequency (%)")
    ax_seq_len_cdf.grid(True, which="both", linewidth=0.5)
    ax_seq_len_cdf.xaxis.set_major_formatter(ScalarFormatter())
    ax_seq_len_cdf.legend()

    _plot_cdf(ax_duration_cdf, durations_rank, "phase1_rank", "C0")
    _plot_cdf(ax_duration_cdf, durations_all, "phase1", "C1")
    ax_duration_cdf.set_xscale("log")
    ax_duration_cdf.set_ylim(0, 100)
    ax_duration_cdf.set_xlabel("Segment Duration (s)")
    ax_duration_cdf.set_ylabel("Cumulative Frequency (%)")
    ax_duration_cdf.grid(True, which="both", linewidth=0.5)
    ax_duration_cdf.xaxis.set_major_formatter(ScalarFormatter())
    ax_duration_cdf.legend()

    def get_ac_dist(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.group_by("aircraft_type")
            .len()
            .with_columns((pl.col("len") / pl.col("len").sum() * 100).alias("percentage"))
            .sort("percentage", descending=True)
        )

    ac_dist_all = get_ac_dist(df_all)
    ac_dist_rank = get_ac_dist(df_rank)

    plot_data = (
        ac_dist_all.select("aircraft_type")
        .join(
            ac_dist_all.select(
                "aircraft_type",
                pl.col("percentage").alias("phase1"),
                pl.col("len").alias("phase1_count"),
            ),
            on="aircraft_type",
            how="left",
        )
        .join(
            ac_dist_rank.select(
                "aircraft_type",
                pl.col("percentage").alias("phase1_rank"),
                pl.col("len").alias("phase1_rank_count"),
            ),
            on="aircraft_type",
            how="left",
        )
        .fill_null(0)
    )

    ac_types = plot_data["aircraft_type"].to_list()
    y = np.arange(len(ac_types))
    bar_height = 0.4

    bars1 = ax_ac_dist.barh(
        y - bar_height / 2,
        plot_data["phase1"],
        height=bar_height,
        label="phase1",
        color="C1",
    )
    bars2 = ax_ac_dist.barh(
        y + bar_height / 2,
        plot_data["phase1_rank"],
        height=bar_height,
        label="phase1_rank",
        color="C0",
    )
    ax_ac_dist.bar_label(
        bars1, labels=plot_data["phase1_count"].cast(pl.Int64), padding=3, color="C1"
    )
    ax_ac_dist.bar_label(
        bars2,
        labels=plot_data["phase1_rank_count"].cast(pl.Int64),
        padding=3,
        color="C0",
    )

    ax_ac_dist.set_ylabel("Aircraft Type")
    ax_ac_dist.set_xlabel("Percentage of Segments (%)")
    ax_ac_dist.set_title("Aircraft Type Distribution")
    ax_ac_dist.set_yticks(y)
    ax_ac_dist.set_yticklabels(ac_types)
    ax_ac_dist.invert_yaxis()
    ax_ac_dist.legend()
    ax_ac_dist.grid(True, axis="x", alpha=0.3)

    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], wspace=0.3)
    ax_seq_len_whisker = fig.add_subplot(gs_bottom[0])
    ax_duration_whisker = fig.add_subplot(gs_bottom[1])

    combined_df = pl.concat(
        [
            df_all.with_columns(pl.lit("phase1").alias("partition")),
            df_rank.with_columns(pl.lit("phase1_rank").alias("partition")),
        ]
    )

    ordered_ac_types = ac_dist_all["aircraft_type"].to_list()
    y_pos = np.arange(len(ordered_ac_types))

    def _plot_whiskers(ax, data_df, value_col, title, xlabel):
        box_width = 0.4
        colors = {"phase1": "C1", "phase1_rank": "C0"}
        positions = {"phase1": y_pos - box_width / 2, "phase1_rank": y_pos + box_width / 2}

        for partition in ["phase1", "phase1_rank"]:
            partition_data = []
            for ac_type in ordered_ac_types:
                series = data_df.filter(
                    (pl.col("aircraft_type") == ac_type) & (pl.col("partition") == partition)
                )[value_col]
                partition_data.append(series.to_numpy())

            bp = ax.boxplot(
                partition_data,
                positions=positions[partition],
                vert=False,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[partition])
                patch.set_alpha(0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(ordered_ac_types)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, axis="x", which="both", alpha=0.3)
        ax.legend(
            [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
            colors.keys(),
        )

    _plot_whiskers(
        ax_seq_len_whisker,
        combined_df,
        "seq_len",
        "Sequence Length Distribution by Aircraft Type",
        "Sequence Length",
    )
    _plot_whiskers(
        ax_duration_whisker,
        combined_df,
        "duration_s",
        "Segment Duration Distribution by Aircraft Type",
        "Segment Duration (s)",
    )

    output_path = PATH_PLOTS_OUTPUT / "segment_distributions.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"wrote plot to {output_path}")


@app.command()
def preprocessed_features_cdf(
    partition: Partition = "phase1",
    split: Split = "train",
):
    import numpy as np
    import polars as pl
    from rich.progress import track

    from microfuel.datasets import preprocessed
    from microfuel.datasets.preprocessed import TrajectoryIterator

    splits = preprocessed.load_splits(partition)
    segment_ids = splits[split]

    iterator = TrajectoryIterator(
        partition=partition,
        segment_ids=segment_ids,
        start_to_end_only=True,
    )

    feature_frames: list[pl.DataFrame] = []
    fuel_kg: list[float] = []
    for trajectory in track(iterator, description=f"loading {split} data for distributions"):
        feature_frames.append(trajectory.features_df)
        fuel_kg.append(trajectory.info["fuel_kg"])

    features_df = pl.concat(feature_frames)
    fuel_kg_np = np.array(fuel_kg)

    num_features = len(preprocessed.STATE_FEATURES)
    fig_cdf, axes_cdf = plt.subplots(2, (num_features + 1 + 1) // 2, figsize=(20, 10))
    axes_cdf: list[Axes] = axes_cdf.flatten()  # type: ignore

    for i, feature in enumerate(preprocessed.STATE_FEATURES):
        ax = axes_cdf[i]
        data = features_df[feature].to_numpy()
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        p1, p5, p50, p95, p99 = np.percentile(data, [1, 5, 50, 95, 99])
        ax.plot(sorted_data, cdf)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("value")
        ax.set_ylabel("cumulative probability")
        ax.set_title(f"{feature}\n{p1=:.3f}, {p5=:.3f}, {p50=:.3f}, {p95=:.3f}, {p99=:.3f}")
        ax.grid(True, alpha=0.3)

    ax_y = axes_cdf[num_features]
    sorted_y = np.sort(fuel_kg_np)
    cdf_y = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
    p1, p5, p50, p95, p99 = np.percentile(fuel_kg_np, [1, 5, 50, 95, 99])
    ax_y.plot(sorted_y, cdf_y, linewidth=2)
    ax_y.set_xlabel("value")
    ax_y.set_ylabel("cumulative probability")
    ax_y.set_title(f"target\n{p1=:.4f}, {p5=:.4f}, {p50=:.4f}, {p95=:.4f}, {p99=:.4f}")
    ax_y.grid(True, alpha=0.3)

    for j in range(num_features + 1, len(axes_cdf)):
        fig_cdf.delaxes(axes_cdf[j])

    output_path_cdf = PATH_PLOTS_OUTPUT / "preprocessed_features_cdf.png"
    fig_cdf.savefig(output_path_cdf, bbox_inches="tight", dpi=300)
    plt.close(fig_cdf)
    logger.info(f"wrote cdf plot to {output_path_cdf}")


#
# train
#


@app.command()
def dataloader(
    partition: str = "phase1",
    split: str = "train",
    batch_size: int = 16,
    num_batches_to_plot: int = 16,
):
    from torch.utils.data import DataLoader

    from microfuel.dataloader import VarlenDataset, collate_fn

    dataset = VarlenDataset(partition=partition, split=split)  # type: ignore
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    fig_batches, axes = plt.subplots(
        4, 4, figsize=(40, 24), sharex=True, sharey=True, layout="tight"
    )
    axes = axes.flatten()
    fig_batches.suptitle(f"sample batches from {partition}/{split} dataloader")

    batches_to_plot = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches_to_plot:
            break
        batches_to_plot.append(batch)

    handles, labels = None, None
    for i, batch in enumerate(batches_to_plot):
        ax = axes[i]
        _plot_varlen_batch(ax, batch)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax.set_title(f"Batch {i + 1}")

    if handles and labels:
        fig_batches.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(labels)
        )

    output_path_batches = PATH_PLOTS_OUTPUT / "dataloader_batches.png"
    fig_batches.savefig(output_path_batches, bbox_inches="tight", dpi=150)
    plt.close(fig_batches)
    logger.info(f"wrote batch plot to {output_path_batches}")


def _plot_varlen_batch(ax: Axes, data: VarlenBatch):
    import matplotlib.colors as colors
    import numpy as np

    from microfuel.datasets import preprocessed

    x = np.arange(data.x.size(0))

    for i, feature in enumerate(preprocessed.STATE_FEATURES):
        ax.scatter(x, data.x[:, i].cpu().numpy(), label=feature, s=0.2)

    for offset in data.cu_seqlens.cpu().numpy():
        ax.axvline(offset, lw=0.5, color="gray")

    y_min, y_max = ax.get_ylim()
    y_pos = y_min + (y_max - y_min) * 0.1

    y_values_log = data.y.cpu().numpy().flatten()
    y_values = np.exp(y_values_log) - 1.0
    norm = colors.Normalize(vmin=y_values.min(), vmax=y_values.max())
    cmap = plt.get_cmap("viridis")

    for i in range(len(data.cu_seqlens) - 1):
        start = data.cu_seqlens[i].item()
        end = data.cu_seqlens[i + 1].item()
        segment_id = data.segment_ids[i].item()
        x_pos = (start + end) / 2
        y = y_values[i]
        color = cmap(norm(y))
        ax.axvspan(start, end, alpha=0.3, color=color)
        ax.text(
            x_pos,
            y_pos,
            f"{segment_id}\n{y:.4f}",
            ha="center",
            va="top",
            rotation=45,
            color=color,
            fontsize=8,
        )


def _plot_scatter(
    ax: Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seq_len: np.ndarray,
    *,
    title: str,
    unit: str,
    norm,
):
    import numpy as np

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    stderr_rmse = (
        np.std((y_true - y_pred) ** 2) / np.sqrt(len(y_true)) / (2 * rmse) if rmse > 0 else 0
    )
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    scatter = ax.scatter(
        y_true, y_pred, alpha=0.3, s=2, linewidth=0, c=seq_len, norm=norm, cmap="viridis"
    )
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="black", lw=0.5)
    ax.set_xlabel(f"Actual ({unit})")
    ax.set_ylabel(f"Predicted ({unit})")
    ax.set_title(f"{title}\nRMSE={rmse:.4f}±{stderr_rmse:.4f}, R²={r2:.4f}")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return scatter


def _plot_mae_vs_duration(
    ax: Axes, duration_s: np.ndarray, mae: np.ndarray, seq_len: np.ndarray, *, unit: str, norm
):
    scatter = ax.scatter(
        duration_s,
        mae,
        s=2,
        linewidth=0,
        alpha=0.5,
        c=seq_len,
        norm=norm,
        cmap="viridis",
    )

    valid_indices = (duration_s > 0) & (mae > 0)

    log_duration = np.log10(duration_s[valid_indices])
    log_mae = np.log10(mae[valid_indices])

    coeffs = np.polyfit(log_duration, log_mae, 1)
    slope, intercept = coeffs

    duration_sorted = np.sort(duration_s[valid_indices])
    trendline = 10 ** (slope * np.log10(duration_sorted) + intercept)

    equation = f"$y = {10**intercept:.4f} x^{{{slope:.4f}}}$"
    ax.plot(duration_sorted, trendline, "k-", linewidth=0.5, label=equation)

    ax.set_xlabel("Segment Duration (s)")
    ax.set_ylabel(f"MAE ({unit})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return scatter


def _plot_dist_cdf(ax: Axes, y_true: np.ndarray, y_pred: np.ndarray, *, unit: str):
    import numpy as np

    sorted_true = np.sort(y_true)
    sorted_pred = np.sort(y_pred)
    cdf_true = np.arange(1, len(sorted_true) + 1) / len(sorted_true) * 100
    cdf_pred = np.arange(1, len(sorted_pred) + 1) / len(sorted_pred) * 100
    ax.plot(sorted_true, cdf_true, lw=0.5, label="Actual")
    ax.plot(sorted_pred, cdf_pred, lw=0.5, label="Predicted")
    ax.set_xlabel(f"Value ({unit})")
    ax.set_ylabel("Cumulative Frequency (%)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")


def _plot_rmse_cdf_by_actype(ax: Axes, df: pl.DataFrame, error_col: str, *, unit: str):
    import numpy as np
    import polars as pl

    df = df.with_columns((pl.col(error_col) ** 2).alias("se"))

    grouped = (
        df.with_columns(pl.col("aircraft_type").cast(pl.Enum(AIRCRAFT_TYPES)))
        .group_by("aircraft_type")
        .agg(
            pl.col(error_col).alias("errors"),
            pl.mean("se").sqrt().alias("rmse"),
            pl.std("se").alias("std_se").fill_null(1),
            pl.len().alias("n"),
        )
        .sort("aircraft_type")
    )

    for row in grouped.iter_rows(named=True):
        ac_type = row["aircraft_type"]
        rmse = row["rmse"]
        stderr_rmse = row["std_se"] / np.sqrt(row["n"]) / (2 * rmse) if rmse > 0 else 0
        errors = np.sort(row["errors"])
        cdf = np.arange(1, len(errors) + 1) / len(errors) * 100

        ax.plot(errors, cdf, lw=0.5, label=f"{ac_type} (RMSE: {rmse:.4f}±{stderr_rmse:.4f})")

    ax.set_xlabel(f"Absolute Error ({unit})")
    ax.set_ylabel("Cumulative Frequency (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_title("CDF of Absolute Error by Aircraft Type")


@app.command()
def predictions(predictions_path: Path, partition: Partition = "phase1"):
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import polars as pl

    preds_df = pl.read_parquet(predictions_path)

    segment_info_df = _load_segment_info(partition).collect()

    plot_df = preds_df.join(segment_info_df, on="segment_id")

    plot_df = plot_df.with_columns(
        (pl.col("y_pred_rate") - pl.col("y_true_rate")).abs().alias("mae_rate"),
        (pl.col("y_pred_kg") - pl.col("y_true_kg")).abs().alias("mae_kg"),
    )

    y_pred_rate, y_true_rate = plot_df.select(["y_pred_rate", "y_true_rate"]).to_numpy().T
    y_pred_kg, y_true_kg = plot_df.select(["y_pred_kg", "y_true_kg"]).to_numpy().T
    duration_s = plot_df["duration_s"].to_numpy()
    mae_rate = plot_df["mae_rate"].to_numpy()
    mae_kg = plot_df["mae_kg"].to_numpy()
    seq_len = plot_df["seq_len"].to_numpy()

    fig = plt.figure(figsize=(20, 32))
    gs = gridspec.GridSpec(5, 2, height_ratios=[1, 1, 1, 1, 0.05], hspace=0.5)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    ax30 = fig.add_subplot(gs[3, 0])
    ax31 = fig.add_subplot(gs[3, 1])
    cbar_ax = fig.add_subplot(gs[4, :])

    norm = mcolors.LogNorm(vmin=max(1, seq_len.min()), vmax=seq_len.max())

    mappable = _plot_scatter(
        ax00, y_true_rate, y_pred_rate, seq_len, title="Avg. Fuel Burn Rate", unit="kg/s", norm=norm
    )
    ax00.set_xlim(1e-2, 1e1)
    ax00.set_ylim(1e-2, 1e1)
    _plot_scatter(
        ax01, y_true_kg, y_pred_kg, seq_len, title="Total Fuel Burn", unit="kg", norm=norm
    )
    ax01.set_xlim(1e0, 1e4)
    ax01.set_ylim(1e0, 1e4)

    _plot_mae_vs_duration(ax10, duration_s, mae_rate, seq_len, unit="kg/s", norm=norm)
    ax10.set_xlim(1e1, 1e4)
    ax10.set_ylim(1e-4, 1e1)
    _plot_mae_vs_duration(ax11, duration_s, mae_kg, seq_len, unit="kg", norm=norm)
    ax11.set_xlim(1e1, 1e4)
    ax11.set_ylim(1e-2, 1e4)

    _plot_dist_cdf(ax20, y_true_rate, y_pred_rate, unit="kg/s")
    ax20.set_xlim(1e-2, 1e1)
    _plot_dist_cdf(ax21, y_true_kg, y_pred_kg, unit="kg")
    ax21.set_xlim(1e0, 1e4)

    _plot_rmse_cdf_by_actype(ax30, plot_df, "mae_rate", unit="kg/s")
    ax30.set_xlim(1e-3, 1e1)
    _plot_rmse_cdf_by_actype(ax31, plot_df, "mae_kg", unit="kg")
    ax31.set_xlim(1e-1, 1e4)

    fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal", label="Sequence Length")

    exp_name = predictions_path.stem.replace("_validation", "").replace("_test", "")
    plot_path = PATH_PLOTS_OUTPUT / "predictions" / f"{exp_name}.png"
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"wrote prediction plot to {plot_path}")


def runs_multi_predictions():
    from microfuel import PATH_PREDICTIONS

    for f in PATH_PREDICTIONS.glob("*.parquet"):
        run_id = f.stem.removeprefix("gdn-all_ac-").removesuffix("_validation")
        pre = "v0.0.9+seed"
        suf = "+cb0.99+dev1"
        if not run_id.startswith(pre) or not run_id.endswith(suf):
            continue
        lf = pl.scan_parquet(f)
        yield run_id.removeprefix(pre).removesuffix(suf), lf


@app.command()
def multi_predictions():
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(12, 35))
    gs = gridspec.GridSpec(7, 1, height_ratios=[1, 1, 1, 1, 2, 2, 0.4], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax_bar = fig.add_subplot(gs[4])
    ax_rmse_vs_freq = fig.add_subplot(gs[5])
    legend_ax = fig.add_subplot(gs[6])
    legend_ax.axis("off")

    segment_info_df = _load_segment_info("phase1").collect()
    segment_id_to_seq_len = {
        row["segment_id"]: row["seq_len"] for row in segment_info_df.iter_rows(named=True)
    }

    handles, labels = [], []
    all_runs_dfs: list[pl.DataFrame] = []
    for run_id, lf in runs_multi_predictions():
        df_with_info = (
            lf.join(segment_info_df.lazy(), on="segment_id", how="left")
            .select(
                pl.col("segment_id"),
                pl.col("duration_s"),
                ((pl.col("y_pred_kg") - pl.col("y_true_kg")) ** 2).alias("se"),
                pl.col("aircraft_type"),
                pl.col("seq_len"),
            )
            .collect()
        )
        all_runs_dfs.append(df_with_info.with_columns(pl.lit(run_id).alias("run_id")))

        rmse = np.sqrt(np.mean(df_with_info["se"].to_numpy()))
        label = f"{run_id} (RMSE: {rmse:.2f})"

        sort_indices = np.argsort(df_with_info["duration_s"].to_numpy())
        durations = df_with_info["duration_s"].to_numpy()[sort_indices]
        squared_errors = df_with_info["se"].to_numpy()[sort_indices]

        cum_durations = np.cumsum(durations)
        cum_se = np.cumsum(squared_errors)

        (line1,) = ax1.plot(cum_durations, cum_se, label=label, linewidth=1)

        seq_lengths = [segment_id_to_seq_len[sid] for sid in df_with_info["segment_id"].to_list()]
        squared_errors_all = df_with_info["se"].to_numpy()

        sort_indices_seq = np.argsort(seq_lengths)
        seq_lengths_sorted = np.array(seq_lengths)[sort_indices_seq]
        squared_errors_sorted = squared_errors_all[sort_indices_seq]

        cum_seq_len = np.cumsum(seq_lengths_sorted)
        cum_se_seq = np.cumsum(squared_errors_sorted)

        ax2.plot(cum_seq_len, cum_se_seq, label=label, lw=1, color=line1.get_color())

        sorted_durations = np.sort(durations)
        cdf_durations = np.arange(1, len(sorted_durations) + 1) / len(sorted_durations) * 100
        ax3.plot(sorted_durations, cdf_durations, label=label, lw=1, color=line1.get_color())

        sorted_seq_len = np.sort(seq_lengths_sorted)
        cdf_seq_len = np.arange(1, len(sorted_seq_len) + 1) / len(sorted_seq_len) * 100
        ax4.plot(sorted_seq_len, cdf_seq_len, label=label, lw=1, color=line1.get_color())

        if len(handles) == 0 or run_id not in labels:
            handles.append(line1)
            labels.append(label)

    df_rank = _load_segment_info("phase1_rank").collect()
    seq_lens_rank = df_rank["seq_len"].to_numpy()
    durations_rank = df_rank["duration_s"].to_numpy()

    sorted_durations_rank = np.sort(durations_rank)
    cdf_durations_rank = (
        np.arange(1, len(sorted_durations_rank) + 1) / len(sorted_durations_rank) * 100
    )
    ax3.plot(
        sorted_durations_rank,
        cdf_durations_rank,
        label="phase1_rank",
        lw=1,
        color="black",
        linestyle="--",
    )
    sorted_seq_len_rank = np.sort(seq_lens_rank)
    cdf_seq_len_rank = np.arange(1, len(sorted_seq_len_rank) + 1) / len(sorted_seq_len_rank) * 100
    (line_rank,) = ax4.plot(
        sorted_seq_len_rank,
        cdf_seq_len_rank,
        label="phase1_rank",
        lw=1,
        color="black",
        linestyle="--",
    )
    handles.append(line_rank)
    labels.append("phase1_rank")

    all_runs_df = pl.concat(all_runs_dfs)
    agg_df = all_runs_df.group_by("run_id", "aircraft_type").agg(
        pl.sum("se").alias("sse"), pl.len().alias("count")
    )
    total_counts_by_run = agg_df.group_by("run_id").agg(pl.sum("count").alias("total_count"))
    agg_df = agg_df.join(total_counts_by_run, on="run_id").with_columns(
        (pl.col("count") / pl.col("total_count")).alias("percentage"),
        (pl.col("sse") / pl.col("count")).sqrt().alias("rmse"),
    )

    def pivot_metric(df: pl.DataFrame, metric: str) -> pl.DataFrame:
        return (
            df.pivot(index="aircraft_type", on="run_id", values=metric)
            .fill_null(0)
            .with_columns(pl.col("aircraft_type").cast(pl.Enum(AIRCRAFT_TYPES)))
            .sort("aircraft_type")
        )

    pivot_sse_df = pivot_metric(agg_df, "sse")
    pivot_counts_df = pivot_metric(agg_df, "count")

    ac_types_in_plot = sorted(agg_df["aircraft_type"].unique().to_list())
    cmap_scatter = plt.get_cmap("tab20")
    colors_scatter = cmap_scatter.colors
    ac_color_map = {
        ac: colors_scatter[i % len(colors_scatter)] for i, ac in enumerate(ac_types_in_plot)
    }

    run_ids_sorted = sorted([c for c in pivot_sse_df.columns if c != "aircraft_type"])
    ac_types = pivot_sse_df["aircraft_type"].to_list()
    x = np.arange(len(run_ids_sorted))
    bottom = np.zeros(len(run_ids_sorted))

    for ac_type in ac_types:
        sse_values = (
            pivot_sse_df.filter(pl.col("aircraft_type") == ac_type)
            .select(run_ids_sorted)
            .to_numpy()
            .flatten()
        )
        count_values = (
            pivot_counts_df.filter(pl.col("aircraft_type") == ac_type)
            .select(run_ids_sorted)
            .to_numpy()
            .flatten()
        )
        color = ac_color_map[ac_type]
        bars = ax_bar.bar(x, sse_values, bottom=bottom, label=ac_type, color=color)

        for j, bar in enumerate(bars):
            sse = sse_values[j]
            count = count_values[j]
            if sse > 0 and bar.get_height() / ax_bar.get_ylim()[1] > 0.015:
                y_pos = bottom[j] + sse / 2
                rgba_color = color
                luminance = 0.299 * rgba_color[0] + 0.587 * rgba_color[1] + 0.114 * rgba_color[2]
                text_color = "white" if luminance < 0.5 else "black"
                ax_bar.text(
                    x[j],
                    y_pos,
                    f"{ac_type}\n{int(count)}",
                    ha="center",
                    va="center",
                    fontsize=4,
                    color=text_color,
                    fontweight="bold",
                )
        bottom += sse_values

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [run_id.replace("v0.0.6+seed", "s") for run_id in run_ids_sorted],
        rotation=45,
        ha="right",
    )
    ax_bar.set_ylabel("Sum of Squared Errors (kg²)")
    ax_bar.set_title("SSE Breakdown by Aircraft Type per Run")
    ax_bar.grid(True, axis="y", alpha=0.3)

    for (ac_type,), group_df in agg_df.sort("aircraft_type").group_by("aircraft_type"):
        sse_vals = group_df["sse"].to_numpy()
        freq_vals = group_df["percentage"].to_numpy()
        ax_rmse_vs_freq.scatter(
            freq_vals,
            sse_vals,
            label=ac_type,
            color=ac_color_map.get(ac_type, "grey"),
            alpha=0.7,
            edgecolors="none",
            s=50,
        )

    ax_rmse_vs_freq.set_xlabel("Frequency of aircraft type in run (fraction)")
    ax_rmse_vs_freq.set_ylabel("Sum of Squared Errors (kg²)")
    ax_rmse_vs_freq.set_title("SSE vs Frequency per Aircraft Type and Run")
    ax_rmse_vs_freq.grid(True, which="both", alpha=0.3)
    ax_rmse_vs_freq.legend(
        title="Aircraft Type", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small"
    )
    ax_rmse_vs_freq.set_xscale("log")

    ax1.set_xlabel("Cumulative Duration (s)")
    ax1.set_ylabel("Cumulative Squared Error (kg²)")
    ax1.grid(True, alpha=0.1, which="both")

    ax2.set_xlabel("Cumulative Sequence Length")
    ax2.set_ylabel("Cumulative Squared Error (kg²)")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.1, which="both")

    ax3.set_xlabel("Segment Duration (s)")
    ax3.set_ylabel("Cumulative Frequency (%)")
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.1, which="both")

    ax4.set_xlabel("Sequence Length")
    ax4.set_ylabel("Cumulative Frequency (%)")
    ax4.set_ylim(0, 100)
    ax4.set_xscale("log")
    ax4.grid(True, alpha=0.1, which="both")

    legend_ax.legend(handles, labels, loc="center", ncol=min(len(labels), 3))

    path_out = PATH_PLOTS_OUTPUT / "multi_predictions.png"
    fig.savefig(path_out, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(f"wrote {path_out}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=False)]
    )
    app()
