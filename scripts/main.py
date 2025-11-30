# NOTE: imports here should be minimal, put heavy imports (torch) inside functions!
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, assert_never, overload

import typer
from rich.logging import RichHandler

from microfuel import PATH_CHECKPOINTS, PATH_DATA_RAW, PATH_PREDICTIONS, Partition, Split

if TYPE_CHECKING:
    from typing import Any

    import polars as pl
    import torch
    from rich.progress import Progress

    from microfuel.model import FuelBurnPredictor, FuelBurnPredictorConfig


logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class EvalResult(NamedTuple):
    rmse_rate: float
    rmse_kg: float
    df: pl.DataFrame


class PredResult(NamedTuple):
    df: pl.DataFrame


@app.command()
def download_raw():
    from microfuel.datasets.raw import download_from_s3, load_config

    config = load_config()
    download_from_s3(config["bucket_access_key"], config["bucket_access_secret"])


@app.command()
def create_actype_enum(partition: Partition = "phase1"):
    from microfuel import AIRCRAFT_TYPES
    from microfuel.datasets import raw

    df = raw.scan_flight_list(partition).collect()
    top_ac_types = (
        df.group_by("aircraft_type").len().sort(by=("len", "aircraft_type"), descending=True)
    )
    total = top_ac_types["len"].sum()
    if list(AIRCRAFT_TYPES) != (expected_ac_types := top_ac_types["aircraft_type"].to_list()):
        logger.warning(
            f"expected `AIRCRAFT_TYPES` enum to be {expected_ac_types}, but got {AIRCRAFT_TYPES}"
        )
    source = "Literal[\n"
    for aircraft_type, count in top_ac_types.iter_rows():
        source += f'    "{aircraft_type}",  # {count / total:.2%}\n'
    source += "]"
    print(source)


@app.command()
def create_splits(
    partition: Partition,
    train_split: float = 0.8,
    seed: int = 25,
):
    from microfuel.datasets.preprocessed import make_splits

    make_splits(partition=partition, train_split=train_split, seed=seed)


@app.command()
def create_dataset(
    partition: Partition,
):
    from microfuel.datasets.preprocessed import make_trajectories

    make_trajectories(partition=partition)


@app.command()
def create_derived_features(
    partition: Partition,
):
    from microfuel.datasets.preprocessed import make_derived_features

    make_derived_features(partition=partition)


@app.command()
def create_stats(
    partition: Partition,
):
    from microfuel.datasets.preprocessed import make_standardisation_stats

    make_standardisation_stats(partition=partition)


# NOTE: not used in training, purely for debugging in `./plots.py`.
@app.command()
def create_segment_info(partition: Partition):
    import polars as pl
    from rich.progress import track

    from microfuel import PATH_PREPROCESSED
    from microfuel.datasets import preprocessed

    description = f"extracting segment info {partition}/all"

    iterator = preprocessed.TrajectoryIterator(partition, start_to_end_only=True)

    segment_stats_data = []
    for trajectory in track(iterator, description=description, total=len(iterator)):
        duration_s = (trajectory.info["end"] - trajectory.info["start"]).total_seconds()
        segment_stats_data.append(
            {
                "segment_id": trajectory.info["idx"],
                "flight_id": trajectory.info["flight_id"],
                "aircraft_type": trajectory.info["aircraft_type"],
                "seq_len": len(trajectory.features_df),
                "duration_s": duration_s,
            }
        )

    df = pl.DataFrame(segment_stats_data)
    output_path = PATH_PREPROCESSED / f"segment_info_{partition}.parquet"
    df.write_parquet(output_path)
    logger.info(f"wrote segment stats to {output_path}")


#
# weather
#

# fmt: off
DEFAULT_LEVELS_BELOW_60K_FT = [
    "70", "100", "125", "150", "175", "200", "225", "250", "300",
    "350", "400", "450", "500", "550", "600", "650", "700", "750",
    "775", "800", "825", "850", "875", "900", "925", "950", "975", "1000"
]
# fmt: on


@app.command()
def download_era5(
    year: str = "2025",
    months: list[str] = ["04", "05", "06", "07", "08", "09", "10"],
    variables: list[str] = ["u_component_of_wind", "v_component_of_wind"],
    levels: list[str] = DEFAULT_LEVELS_BELOW_60K_FT,
    base_url: str = "gs://gcp-public-data-arco-era5/raw/date-variable-pressure_level",
    path_base: Path = PATH_DATA_RAW / "era5",
):
    import calendar
    import subprocess

    logger.info(f"starting download for {year} months: {months}")
    logger.info(f"variables: {variables}")
    logger.info(f"levels: {len(levels)} levels")

    for month in months:
        _, days_in_month = calendar.monthrange(int(year), int(month))

        for day_int in range(1, days_in_month + 1):
            day = f"{day_int:02d}"
            logger.info(f"processing {year}-{month}-{day}...")

            for var in variables:
                dest_dir = path_base / year / month / day / var
                dest_dir.mkdir(parents=True, exist_ok=True)

                existing = list(dest_dir.glob("*.nc"))
                if len(existing) == len(levels):
                    logger.info(f"skipping {dest_dir}, already populated")
                    continue

                url_list = []
                for lvl in levels:
                    url = f"{base_url}/{year}/{month}/{day}/{var}/{lvl}.nc"
                    url_list.append(url)

                process = subprocess.Popen(
                    ["gcloud", "storage", "cp", "-I", str(dest_dir)],
                    stdin=subprocess.PIPE,
                    text=True,
                )
                process.communicate(input="\n".join(url_list))

                if process.returncode != 0:
                    logger.error(f"failed to download batch for {year}-{month}-{day} {var}")

    logger.info("download complete")


@app.command()
def create_era5(
    partition: Partition,
):
    from microfuel.datasets.preprocessed import make_era5

    make_era5(partition=partition)


@dataclass
class TrainConfig:
    partition: Partition
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    warmup_steps: int
    seed: int
    model_config: FuelBurnPredictorConfig
    project_name: str
    exp_name: str


@dataclass
class Checkpoint:
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scaler_state_dict: dict[str, Any]
    global_step: int
    train_config: TrainConfig


@app.command()
def train(
    partition: Partition = "phase1",
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 4e-4,
    hidden_size: int = 32,
    num_heads: int = 2,
    num_layers: int = 3,
    aircraft_embedding_dim: int = 8,
    pooler_mode: Literal["mean", "last"] = "last",
    beta: Annotated[
        float,
        typer.Option(help="Hyperparameter for CB Loss. 0.0 means no reweighting."),
    ] = 0.99,
    *,
    project_name: str = "prc25-multiac",
    exp_name: str = "gdn-all_ac-v0.0.9",
    resume_from: Annotated[
        Path | None, typer.Option(help="Path to checkpoint to resume training from.")
    ] = None,
    loss_type: Literal["rmse_rate", "rmse_kg", "rmse_kg2"] = "rmse_kg",
    evaluate_best: Annotated[
        bool, typer.Option(help="Evaluate the best model on the validation set after training.")
    ] = True,
    weight_decay: float = 0.1,
    warmup_steps: int = 250,
    seed: int = 13,
    evaluate_every: Annotated[int, typer.Option(help="Evaluate the model every n steps.")] = 500,
):
    import math

    import torch
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader

    from microfuel.hacks import install_optimized_kernels_

    install_optimized_kernels_()
    torch.manual_seed(seed)
    import polars as pl

    import wandb
    from microfuel.dataloader import VarlenDataset, collate_fn
    from microfuel.model import FuelBurnPredictor, FuelBurnPredictorConfig

    # NOTE: loading this takes 16GB of RAM on start, but drops to ~4GB
    train_dataset = VarlenDataset(partition=partition, split="train")
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataset = VarlenDataset(partition=partition, split="validation")
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    idx_to_ac_type = {i: ac for ac, i in train_dataset.ac_type_vocab.items()}
    weights = None
    if beta > 0.0:
        class_counts = train_dataset.class_counts
        effective_num = 1.0 - torch.pow(beta, class_counts.to(torch.float32))
        weights = (1.0 - beta) / effective_num
        logger.info(f"using cb loss with {beta=} on {class_counts=}: {weights=}")

    model_cfg = FuelBurnPredictorConfig(
        input_dim=train_dataset.all_features.shape[1],
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_aircraft_types=len(train_dataset.ac_type_vocab),
        aircraft_embedding_dim=aircraft_embedding_dim,
        num_layers=num_layers,
        pooler_mode=pooler_mode,
    )
    cfg = TrainConfig(
        partition=partition,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        seed=seed,
        model_config=model_cfg,
        project_name=project_name,
        exp_name=exp_name,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    checkpoint: Checkpoint | None = None
    if resume_from and resume_from.exists():
        logger.info(f"loading checkpoint from {resume_from}")
        with torch.serialization.safe_globals([Checkpoint, TrainConfig, FuelBurnPredictorConfig]):
            checkpoint = torch.load(resume_from, map_location=device)

    exp_checkpoint_dir = PATH_CHECKPOINTS / cfg.exp_name
    exp_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model = FuelBurnPredictor(cfg.model_config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model architecture ({total_params=:,}):")

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.dim() < 2
            or name.endswith(".bias")
            or (hasattr(param, "_no_weight_decay") and param._no_weight_decay)  # type: ignore
        ):
            no_decay_params.append(param)
            logger.info(f"  (no decay) {name:<27} {tuple(param.shape)!s:<15} {param.numel():,}")
        else:
            decay_params.append(param)
            logger.info(f"  (decay)    {name:<27} {tuple(param.shape)!s:<15} {param.numel():,}")

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.lr)
    criterion = torch.nn.MSELoss(reduction="none")

    total_training_steps = len(train_dataloader) * cfg.epochs

    def get_lr_scheduler(optimizer, warmup_steps, total_training_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_training_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, total_training_steps)

    wandb.init(project=cfg.project_name, name=cfg.exp_name, config=asdict(cfg))
    wandb.watch(model, log="all", log_freq=100)

    model.to(device)
    scaler = torch.amp.GradScaler(device=device, enabled=(device == "cuda"))

    if checkpoint:
        # NOTE: assuming reloading from checkpoint to be used for finetuning
        # so we do not restore optimiser and scaler states!
        model.load_state_dict(checkpoint.model_state_dict)

    global_step = 0
    best_val_rmse = float("inf")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        for epoch in range(cfg.epochs):
            model.train()
            train_loss_total = 0.0
            train_samples = 0

            train_task = progress.add_task(
                f"Epoch {epoch + 1}/{cfg.epochs}", total=len(train_dataloader)
            )
            for data in train_dataloader:
                x_flight: torch.Tensor = data.x_flight.to(device)
                cu_seqlens_flight: torch.Tensor = data.cu_seqlens_flight.to(device)
                x_segment: torch.Tensor = data.x_segment.to(device)
                cu_seqlens_segment: torch.Tensor = data.cu_seqlens_segment.to(device)
                y_log: torch.Tensor = data.y.to(device)
                aircraft_type_idx: torch.Tensor = data.aircraft_type_idx.to(device)
                durations: torch.Tensor = data.durations.to(device)

                optimizer.zero_grad()

                with torch.autocast(
                    device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")
                ):
                    y_pred_log = model(
                        x_flight,
                        cu_seqlens_flight,
                        x_segment,
                        cu_seqlens_segment,
                        aircraft_type_idx,
                    )
                    loss_per_sample = criterion(y_pred_log, y_log)

                    if weights is not None:
                        weights = weights.to(device)
                        batch_weights = weights[aircraft_type_idx]
                        loss_per_sample = loss_per_sample * batch_weights.unsqueeze(1)

                    if loss_type == "rmse_rate":
                        loss = loss_per_sample.mean()
                    elif loss_type == "rmse_kg":
                        loss = (loss_per_sample * durations.unsqueeze(1)).mean()
                    elif loss_type == "rmse_kg2":  # very unstable!
                        loss = (loss_per_sample * durations.unsqueeze(1) ** 2).mean()
                    else:
                        assert_never(loss_type)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                if global_step > 0 and global_step % evaluate_every == 0:
                    eval_result = _run_inference(
                        model, val_dataloader, device, progress, "validating", is_eval=True
                    )

                    per_ac_rmse_df = (
                        eval_result.df.with_columns(
                            ((pl.col("y_pred_kg") - pl.col("y_true_kg")) ** 2).alias("se_kg")
                        )
                        .group_by("aircraft_type_idx")
                        .agg(pl.mean("se_kg").sqrt().alias("rmse_kg"))
                    )
                    wandb_per_ac_metrics = {
                        f"rmse_kg_val/{idx_to_ac_type[row['aircraft_type_idx']]}": row["rmse_kg"]
                        for row in per_ac_rmse_df.iter_rows(named=True)
                    }
                    wandb_log_data = {
                        "rmse_rate_val": eval_result.rmse_rate,
                        "rmse_kg_val": eval_result.rmse_kg,
                    }
                    wandb_log_data.update(wandb_per_ac_metrics)
                    wandb.log(
                        wandb_log_data,
                        step=global_step,
                    )
                    if eval_result.rmse_kg < best_val_rmse:
                        best_val_rmse = eval_result.rmse_kg
                        checkpoint_path = (
                            exp_checkpoint_dir / f"step{global_step:05}_{best_val_rmse:.2f}.pt"
                        )
                        checkpoint = Checkpoint(
                            model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            scaler_state_dict=scaler.state_dict(),
                            global_step=global_step,
                            train_config=cfg,
                        )
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(
                            f"wrote {checkpoint_path}: best {best_val_rmse=:.2f} ({eval_result.rmse_rate=:.4f})"
                        )
                    model.train()

                batch_size = y_log.size(0)
                train_loss_total += loss.item() * batch_size
                train_samples += batch_size
                global_step += 1

                with torch.no_grad():
                    y_pred_rate_orig = torch.expm1(y_pred_log.detach())
                    y_true_rate_orig = torch.expm1(y_log.detach())
                    rmse_rate_orig = torch.sqrt(
                        torch.nn.functional.mse_loss(y_pred_rate_orig, y_true_rate_orig)
                    ).item()

                    y_pred_kg = y_pred_rate_orig * durations.unsqueeze(1)
                    y_true_kg = y_true_rate_orig * durations.unsqueeze(1)
                    rmse_kg = torch.sqrt(torch.nn.functional.mse_loss(y_pred_kg, y_true_kg)).item()
                wandb.log(
                    {
                        "lr": scheduler.get_last_lr()[0],
                        "rmse_rate_train": rmse_rate_orig,
                        "rmse_kg_train": rmse_kg,
                        "gradient_norm": total_norm.item(),
                    },
                    step=global_step,
                )

                progress.update(
                    train_task,
                    advance=1,
                    description=(f"train epoch {epoch + 1}/{cfg.epochs} rmse_kg={rmse_kg:.2f}"),
                )
            progress.remove_task(train_task)

    wandb.finish()

    if evaluate_best:
        logger.info("evaluating best model on validation set")
        best_val, best_checkpoint_path = float("inf"), None
        for f in exp_checkpoint_dir.iterdir():
            if f.suffix != ".pt" or (val := float(f.stem.split("_")[1])) >= best_val:
                continue
            best_val = val
            best_checkpoint_path = f
        if best_checkpoint_path is None:
            logger.warning("no best model checkpoint found to evaluate.")
            return
        evaluate(best_checkpoint_path, partition, "validation", batch_size)


@overload
def _run_inference(
    model: FuelBurnPredictor,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    progress: Progress,
    description: str,
    *,
    is_eval: Literal[True],
) -> EvalResult: ...


@overload
def _run_inference(
    model: FuelBurnPredictor,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    progress: Progress,
    description: str,
    *,
    is_eval: Literal[False],
) -> PredResult: ...


def _run_inference(
    model: FuelBurnPredictor,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    progress: Progress,
    description: str,
    *,
    is_eval: bool,
) -> EvalResult | PredResult:
    import polars as pl
    import torch

    model.eval()
    all_preds_rate, all_trues_rate, all_segment_ids, all_durations = [], [], [], []
    all_aircraft_type_idxs = []  # needed for wandb logging per ac type

    with torch.no_grad():
        task = progress.add_task(description, total=len(dataloader))
        for data in dataloader:
            (
                x_flight,
                cu_seqlens_flight,
                x_segment,
                cu_seqlens_segment,
                y_log,
                segment_ids,
                aircraft_type_idx,
                durations,
            ) = (
                data.x_flight.to(device),
                data.cu_seqlens_flight.to(device),
                data.x_segment.to(device),
                data.cu_seqlens_segment.to(device),
                data.y.to(device),
                data.segment_ids.cpu(),
                data.aircraft_type_idx.to(device),
                data.durations.cpu(),
            )
            with torch.autocast(
                device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")
            ):
                y_pred_log = model(
                    x_flight,
                    cu_seqlens_flight,
                    x_segment,
                    cu_seqlens_segment,
                    aircraft_type_idx,
                )

            y_pred_rate = torch.expm1(y_pred_log)
            all_preds_rate.append(y_pred_rate.cpu())
            all_segment_ids.append(segment_ids)
            all_durations.append(durations)
            all_aircraft_type_idxs.append(aircraft_type_idx.cpu())

            if is_eval:
                y_true_rate = torch.expm1(y_log)
                all_trues_rate.append(y_true_rate.cpu())

            progress.update(task, advance=1)
        progress.remove_task(task)

    preds_rate_tensor = torch.cat(all_preds_rate).flatten()
    segment_ids_tensor = torch.cat(all_segment_ids).flatten()
    durations_tensor = torch.cat(all_durations).flatten()
    aircraft_type_idx_tensor = torch.cat(all_aircraft_type_idxs).flatten()

    if is_eval:
        trues_rate_tensor = torch.cat(all_trues_rate).flatten()
        df = pl.DataFrame(
            {
                "segment_id": segment_ids_tensor.numpy(),
                "duration_s": durations_tensor.numpy(),
                "y_true_rate": trues_rate_tensor.numpy(),
                "y_pred_rate": preds_rate_tensor.to(torch.float32).numpy(),
                "aircraft_type_idx": aircraft_type_idx_tensor.numpy(),
            }
        )
        df = df.with_columns(
            (pl.col("y_true_rate") * pl.col("duration_s")).alias("y_true_kg"),
            (pl.col("y_pred_rate") * pl.col("duration_s")).alias("y_pred_kg"),
        )

        rmse_rate = torch.sqrt(torch.mean((preds_rate_tensor - trues_rate_tensor) ** 2)).item()

        preds_kg = preds_rate_tensor * durations_tensor
        trues_kg = trues_rate_tensor * durations_tensor
        rmse_kg = torch.sqrt(torch.mean((preds_kg - trues_kg) ** 2)).item()

        return EvalResult(rmse_rate=rmse_rate, rmse_kg=rmse_kg, df=df)
    else:
        preds_kg_tensor = preds_rate_tensor * durations_tensor
        df = pl.DataFrame(
            {
                "idx": segment_ids_tensor.numpy(),
                "fuel_kg": preds_kg_tensor.to(torch.float32).numpy(),
            }
        )
        return PredResult(df=df)


@app.command()
def evaluate(
    checkpoint_path: Path,
    partition: Partition = "phase1",
    split: Split = "validation",
    batch_size: int = 64,
    *,
    for_submission: bool = False,
):
    import polars as pl
    import torch
    from rich.progress import Progress
    from torch.utils.data import DataLoader

    from microfuel.datasets import raw
    from microfuel.hacks import install_optimized_kernels_

    install_optimized_kernels_()
    from microfuel.dataloader import VarlenDataset, collate_fn
    from microfuel.model import FuelBurnPredictor, FuelBurnPredictorConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    with torch.serialization.safe_globals([Checkpoint, TrainConfig, FuelBurnPredictorConfig]):
        checkpoint: Checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.train_config.model_config
    model = FuelBurnPredictor(model_config)
    model.load_state_dict(checkpoint.model_state_dict)
    model.to(device)

    dataset_split = None if for_submission else split
    if for_submission and "rank" not in partition:
        logger.warning(
            f"generating submission for partition `{partition}` which is not a ranking partition!"
        )

    dataset = VarlenDataset(partition=partition, split=dataset_split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with Progress() as progress:
        if for_submission:
            pred_result = _run_inference(
                model, dataloader, device, progress, f"predicting on {partition=}", is_eval=False
            )
            df_preds = pred_result.df
            df_template = (
                raw.scan_fuel(partition).select("idx", "flight_id", "start", "end").collect()
            )
            df = df_template.join(df_preds, on="idx", how="left").with_columns(pl.col("fuel_kg"))
            df = df.select("idx", "flight_id", "start", "end", "fuel_kg")
        else:
            eval_result = _run_inference(
                model, dataloader, device, progress, f"evaluating on {split=}", is_eval=True
            )
            df = eval_result.df
            logger.info(
                f"final rmse on {split=}:"
                f" rate={eval_result.rmse_rate:.4f} kg/s,"
                f" total={eval_result.rmse_kg:.2f} kg"
            )

    exp_name = checkpoint_path.parent.name
    PATH_PREDICTIONS.mkdir(exist_ok=True, parents=True)

    if for_submission:
        output_path = PATH_PREDICTIONS / f"{exp_name}_{partition}_submission.parquet"
    else:
        output_path = PATH_PREDICTIONS / f"{exp_name}_{split}.parquet"

    df.write_parquet(output_path)
    logger.info(f"wrote predictions to {output_path}")


@app.command()
def submit(
    predictions_path: Annotated[
        Path, typer.Argument(help="Path to the prediction parquet file generated by `evaluate`.")
    ],
    version: Annotated[
        int | None, typer.Option("--version", "-v", help="Submission version number.")
    ] = None,
    final: bool = False,
):
    import os

    import polars as pl

    from microfuel.datasets.raw import load_config, setup_mc_alias

    df = pl.read_parquet(predictions_path)
    config = load_config()
    team_name = config.get("team_name")
    if version and final:
        raise typer.BadParameter("cannot specify both --version and --final")
    if not version and not final:
        raise typer.BadParameter("must specify either --version or --final")
    suffix = "_final" if final else f"_v{version}"
    final_filename = f"{team_name}{suffix}.parquet"

    alias_name = "prc25"
    bucket_name = f"prc-2025-{team_name}"

    remote_path = f"{alias_name}/{bucket_name}/{final_filename}"
    cmd = f"mc cp {predictions_path} {remote_path}"

    print(df.head())
    print(df.select("fuel_kg").describe())
    if not typer.confirm(f"execute {cmd}?"):
        raise typer.Exit()

    setup_mc_alias(
        config["bucket_access_key"], config["bucket_access_secret"], alias_name=alias_name
    )
    return os.system(cmd)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=False)]
    )
    app()
