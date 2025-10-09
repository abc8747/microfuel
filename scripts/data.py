import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import typer
from matplotlib.axes import Axes
from rich.logging import RichHandler
from rich.progress import track

from prc25 import PATH_DATA
from prc25.datasets import raw
from prc25.plot import MPL

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True)


@app.command()
def download_raw():
    from prc25.datasets.raw import download_from_s3, load_config

    config = load_config()
    download_from_s3(config["bucket_access_key"], config["bucket_access_secret"])


#
# exploratory data analysis
#

eda = typer.Typer(no_args_is_help=True)
app.add_typer(eda, name="eda", help="exploratory data analysis")

PATH_EDA_OUTPUT = PATH_DATA / "eda"


@eda.command()
def flight_duration() -> None:
    fig = MPL.default_fig(figsize=(8, 12))
    ax = fig.add_subplot(111)
    df = (
        raw.load_flight_list("train")
        .with_columns(
            duration_hours=(pl.col("landed") - pl.col("takeoff")).dt.total_seconds() / 3600
        )
        .collect()
    )
    total = len(df)

    top_ac_types = df.group_by("aircraft_type").len().sort("len")

    data = []
    labels = []
    for row in top_ac_types.iter_rows(named=True):
        ac_type = row["aircraft_type"]
        count = row["len"]
        pct = (count / total) * 100
        durations = df.filter(pl.col("aircraft_type") == ac_type)["duration_hours"].to_list()
        data.append(durations)
        labels.append(f"{ac_type} ({pct:.2f}%)")

    ax.violinplot(data, positions=range(len(data)), showmeans=True, vert=False)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylabel("Aircraft Type")
    ax.set_xlabel("Flight Duration (hours)")
    ax.grid(True, alpha=0.6, axis="x")

    output_path = PATH_EDA_OUTPUT / "flight_duration.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {output_path}")


@eda.command()
def od_graph() -> None:
    fig = MPL.default_fig(figsize=(13, 12))
    ax = fig.add_subplot(111)
    df = raw.load_flight_list("train").collect()

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

    output_path = PATH_EDA_OUTPUT / "od_graph.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {output_path}")


@eda.command()
def speed_alt_fuel_burn(max_points_per_actype: float | None = 1000) -> None:
    """This is a very slow plot: takes ~1 minute"""
    fig = MPL.default_fig(figsize=(20, 20))

    flight_list_lf = raw.load_flight_list("train")

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
        raw.load_fuel_data("train")
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
        ax = axes.flatten()[i]
        data_subset = plot_df.filter(pl.col("aircraft_type") == ac_type)

        p_lower = max(0, data_subset["avg_fuel_burn_rate_kg_s"].quantile(0.01) or 0)
        p_upper = min(data_subset["avg_fuel_burn_rate_kg_s"].quantile(0.95) or 10, 10)

        for row in track(
            data_subset.iter_rows(named=True),
            total=data_subset.height,
            description=ac_type,
        ):
            traj_lf = raw.load_trajectory(row["flight_id"], "train")
            if traj_lf is None:
                continue

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

    output_path = PATH_EDA_OUTPUT / "speed_alt_fuel_burn.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    logger.info(f"wrote {output_path}")


LBS_TO_KG = 0.45359237


@eda.command()
def fuel_quantisation(tolerance: float = 0.01) -> None:
    fuel_lf = raw.load_fuel_data("train")

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

    colors = {"metric": MPL.C[1], "imperial": MPL.C[2], "neither": MPL.C[3]}
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

    output_path = PATH_EDA_OUTPUT / "fuel_quantisation.png"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"wrote {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=False)]
    )
    app()
