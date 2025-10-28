from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import typer
from matplotlib.ticker import ScalarFormatter
from rich.logging import RichHandler
from rich.progress import track

import prc25.plot as p
from prc25 import PATH_PLOTS_OUTPUT, PATH_PREPROCESSED, Partition, Split
from prc25.datasets import raw

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from prc25.dataloader import VarlenBatch

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

    top_ac_types = df.group_by("aircraft_type").len().sort("len", descending=True)
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
def od_graph() -> None:
    fig = p.default_fig(figsize=(13, 12))
    ax = fig.add_subplot(111)
    df = raw.scan_flight_list("phase1").collect()

    top_routes = df.group_by(["origin_icao", "destination_icao"]).len().sort("len", descending=True)

    graph = nx.DiGraph()

    for row in top_routes.iter_rows(named=True):
        graph.add_edge(row["origin_icao"], row["destination_icao"], weight=row["len"])

    degrees: list[int] = [graph.degree(node) for node in graph.nodes()]  # type: ignore
    node_sizes = [min(d * 50, 500) for d in degrees]

    min_degree = min(degrees)
    max_degree = max(degrees)
    font_sizes = {
        node: 2 + (d - min_degree) / (max_degree - min_degree) * 5
        for node, d in zip(graph.nodes(), degrees)
    }
    labels = {node: f"{node}\n({d})" for node, d in zip(graph.nodes(), degrees)}

    normalized_degrees = {
        node: (d - min_degree) / (max_degree - min_degree)
        for node, d in zip(graph.nodes(), degrees)
    }
    font_colors = {
        node: "black" if norm_d > 0.5 else "white" for node, norm_d in normalized_degrees.items()
    }

    pos = nx.spring_layout(graph, k=15, iterations=1000, seed=13, scale=2)

    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color=degrees,  # type: ignore
        cmap="viridis",
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels=labels,
        font_size=font_sizes,  # type: ignore
        font_color=font_colors,  # type: ignore
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph, pos, arrows=True, arrowsize=10, edge_color="gray", width=1, alpha=0.5, ax=ax
    )

    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label("Node Degree")

    ax.axis("off")

    output_path = PATH_PLOTS_OUTPUT / "od_graph.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {output_path}")


@app.command()
def speed_alt_fuel_burn(max_points_per_actype: float | None = 1000) -> None:
    """This is a very slow plot: takes ~1 minute"""
    fig = p.default_fig(figsize=(20, 20))

    flight_list_lf = raw.scan_flight_list("phase1")

    top_ac_types = (
        flight_list_lf.group_by("aircraft_type")
        .len()
        .sort("len", descending=True)
        .head(16)
        .collect()["aircraft_type"]
        .to_list()
    )

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


def _plot_cdf(ax: Axes, data: pl.Series, label: str, color: str) -> None:
    sorted_data = np.sort(data.to_numpy())
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

    _plot_cdf(ax, all_gaps, "time gaps (all)", "C0")
    _plot_cdf(ax, acars_gaps, "time gaps (acars)", "C2")
    _plot_cdf(ax, segment_lengths, "segment lengths (fuel data)", "C3")

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


#
# preprocessed
#


@app.command()
def seq_len_cdf(partition: Partition = "phase1") -> None:
    from prc25 import SPLITS
    from prc25.datasets import preprocessed

    segment_lengths = []
    for split in SPLITS:
        traj_lf = pl.scan_parquet(PATH_PREPROCESSED / f"trajectories_{partition}_{split}.parquet")
        flight_ids = traj_lf.select("flight_id").unique().collect()["flight_id"].to_list()

        for trajectory in track(
            preprocessed.TrajectoryIterator(
                partition, split, flight_ids=flight_ids, start_to_end_only=True
            ),
            description=f"collecting segment lengths for {split}",
        ):
            segment_lengths.append(len(trajectory.features_df))

    fig = p.default_fig(figsize=(16, 8))
    ax = fig.add_subplot(111)
    _plot_cdf(ax, pl.Series(segment_lengths), "sequence length", "C0")

    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Cumulative Frequency (%)")
    ax.grid(True, which="both", linewidth=0.5)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()

    output_path = PATH_PLOTS_OUTPUT / "seq_len_cdf.pdf"
    fig.savefig(output_path, bbox_inches="tight")
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

    from prc25.datasets import preprocessed
    from prc25.datasets.preprocessed import TrajectoryIterator

    traj_lf = pl.scan_parquet(PATH_PREPROCESSED / f"trajectories_{partition}_{split}.parquet")
    flight_ids = traj_lf.select("flight_id").unique().collect()["flight_id"].to_list()

    iterator = TrajectoryIterator(
        partition=partition,
        split=split,
        flight_ids=flight_ids,
        start_to_end_only=True,
    )

    feature_frames: list[pl.DataFrame] = []
    fuel_kg: list[float] = []
    for trajectory in track(iterator, description=f"loading {split} data for distributions"):
        feature_frames.append(trajectory.features_df)
        fuel_kg.append(trajectory.info["fuel_kg"])

    features_df = pl.concat(feature_frames)
    fuel_kg_np = np.array(fuel_kg)

    num_features = len(preprocessed.TRAJECTORY_FEATURES)
    fig_cdf, axes_cdf = plt.subplots(2, (num_features + 1 + 1) // 2, figsize=(20, 10))
    axes_cdf: list[Axes] = axes_cdf.flatten()  # type: ignore

    for i, feature in enumerate(preprocessed.TRAJECTORY_FEATURES):
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


@app.command()
def preprocessed_trajectories(
    partition: Partition = "phase1",
    num_flights: int = 100,
) -> None:
    output_dir = PATH_PLOTS_OUTPUT / "preprocessed_trajectories"
    output_dir.mkdir(exist_ok=True, parents=True)

    preprocessed_lf = pl.scan_parquet(PATH_PREPROCESSED / f"trajectories_{partition}.parquet")
    flight_list_lf = raw.scan_flight_list(partition)
    fuel_lf = raw.scan_fuel(partition)

    flight_ids = (
        preprocessed_lf.select("flight_id")
        .unique()
        .sort("flight_id")
        .head(num_flights)
        .collect()
        .to_series()
        .to_list()
    )

    segments_lf = (
        flight_list_lf.select("flight_id", "takeoff")
        .filter(pl.col("flight_id").is_in(flight_ids))
        .join(fuel_lf, on="flight_id")
        .with_columns(
            start_s=(pl.col("start") - pl.col("takeoff")).dt.total_seconds(fractional=True),
            end_s=(pl.col("end") - pl.col("takeoff")).dt.total_seconds(fractional=True),
        )
    )

    for flight_id in track(flight_ids, description="plotting trajectories"):
        traj_df = preprocessed_lf.filter(pl.col("flight_id") == flight_id).collect()
        segments_df = segments_lf.filter(pl.col("flight_id") == flight_id).collect()

        fig = p.default_fig(figsize=(20, 8))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.04))

        time = traj_df["time_since_takeoff"]
        p1 = ax1.scatter(time, traj_df["altitude"], s=1, color="C0", label="Altitude")
        p2 = ax2.scatter(time, traj_df["groundspeed"], s=1, color="C1", label="Groundspeed")
        p3 = ax3.scatter(time, traj_df["vertical_rate"], s=1, color="C2", label="Vertical Rate")

        ax1.set_xlabel("Time Since Takeoff (s)")
        ax1.set_ylabel("Altitude (m)", color="C0")
        ax2.set_ylabel("Groundspeed (m/s)", color="C1")
        ax3.set_ylabel("Vertical Rate (m/s)", color="C2")

        ax1.tick_params(axis="y", labelcolor="C0")
        ax2.tick_params(axis="y", labelcolor="C1")
        ax3.tick_params(axis="y", labelcolor="C2")

        for i, row in enumerate(segments_df.iter_rows(named=True)):
            start, end = row["start_s"], row["end_s"]
            color = f"C{(i + 3) % 10}"
            ax1.axvspan(start, end, alpha=0.2, color=color)

            segment_traj = traj_df.filter(pl.col("time_since_takeoff").is_between(start, end))
            duration_s = end - start
            label_text = f"{row['idx']}: {segment_traj.height}/{duration_s:.0f}s"

            ax1.text(
                (start + end) / 2,
                ax1.get_ylim()[1] * 0.15,
                label_text,
                ha="center",
                va="top",
                rotation=45,
                color=color,
            )

        fig.suptitle(f"flight id: {flight_id}")
        fig.legend(handles=[p1, p2, p3], loc="upper right")
        output_path = output_dir / f"{flight_id}.png"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        logger.info(f"wrote {output_path}")
        plt.close(fig)


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

    from prc25.dataloader import VarlenDataset, collate_fn

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

    from prc25.datasets import preprocessed

    x = np.arange(data.x.size(0))

    for i, feature in enumerate(preprocessed.TRAJECTORY_FEATURES):
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
    ax.plot(sorted_true, cdf_true, linewidth=2, label="Actual")
    ax.plot(sorted_pred, cdf_pred, linewidth=2, label="Predicted")
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
        df.group_by("aircraft_type")
        .agg(
            pl.col(error_col).alias("errors"),
            pl.mean("se").sqrt().alias("rmse"),
            pl.std("se").alias("std_se"),
            pl.len().alias("n"),
        )
        .sort("rmse")
    )

    for row in grouped.iter_rows(named=True):
        ac_type = row["aircraft_type"]
        rmse = row["rmse"]
        stderr_rmse = row["std_se"] / np.sqrt(row["n"]) / (2 * rmse) if rmse > 0 else 0
        errors = np.sort(row["errors"])
        cdf = np.arange(1, len(errors) + 1) / len(errors) * 100

        ax.plot(errors, cdf, linewidth=2, label=f"{ac_type} (RMSE: {rmse:.4f}±{stderr_rmse:.4f})")

    ax.set_xlabel(f"Absolute Error ({unit})")
    ax.set_ylabel("Cumulative Frequency (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_title("CDF of Absolute Error by Aircraft Type")


@app.command()
def predictions(
    predictions_path: Path,
    partition: Partition = "phase1",
    split: Split = "validation",
):
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import polars as pl

    from prc25.datasets import raw
    from prc25.datasets.preprocessed import TrajectoryIterator

    preds_df = pl.read_parquet(predictions_path)

    fuel_lf = raw.scan_fuel(partition)
    flight_ids = (
        preds_df.lazy()
        .join(fuel_lf, left_on="segment_id", right_on="idx")
        .select("flight_id")
        .unique()
        .collect()["flight_id"]
        .to_list()
    )

    segment_info_data = []
    trajectory_iterator = TrajectoryIterator(
        partition=partition, split=split, flight_ids=flight_ids, start_to_end_only=True
    )
    for segment in track(trajectory_iterator, description="gathering segment metadata"):
        duration_s = (segment.info["end"] - segment.info["start"]).total_seconds()
        seq_len = len(segment.features_df)
        segment_info_data.append(
            {
                "idx": segment.info["idx"],
                "duration_s": duration_s,
                "seq_len": seq_len,
                "aircraft_type": segment.info["aircraft_type"],
            }
        )
    segment_info_df = pl.DataFrame(segment_info_data)

    plot_df = preds_df.join(segment_info_df, left_on="segment_id", right_on="idx")

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
    ax31.set_xlim(1e0, 1e4)

    fig.colorbar(mappable, cax=cbar_ax, orientation="horizontal", label="Sequence Length")

    exp_name = predictions_path.stem.replace("_validation", "").replace("_test", "")
    plot_path = PATH_PLOTS_OUTPUT / "predictions" / f"{exp_name}.png"
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"wrote prediction plot to {plot_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=False)]
    )
    app()
