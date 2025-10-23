# NOTE: imports here should be minimal, put heavy imports (torch) inside functions!
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.logging import RichHandler

from prc25 import PATH_CHECKPOINTS, PATH_PREDICTIONS, Partition, Split

if TYPE_CHECKING:
    from typing import Any

    import polars as pl
    import torch
    from rich.progress import Progress

    from prc25.model import FuelBurnPredictor, FuelBurnPredictorConfig


logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.command()
def download_raw():
    from prc25.datasets.raw import download_from_s3, load_config

    config = load_config()
    download_from_s3(config["bucket_access_key"], config["bucket_access_secret"])


@app.command()
def create_dataset(
    partition: Partition,
):
    from prc25.datasets.preprocessed import make_trajectories

    make_trajectories(partition=partition)


@app.command()
def create_stats(
    partition: Partition,
):
    from prc25.datasets.preprocessed import make_standardisation_stats

    make_standardisation_stats(partition=partition)


@dataclass
class TrainConfig:
    partition: Partition
    batch_size: int
    epochs: int
    lr: float
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
    epochs: int = 10,
    lr: float = 4e-4,
    hidden_size: int = 32,
    num_heads: int = 2,
    aircraft_embedding_dim: int = 8,
    project_name: str = "prc25-multiac",
    exp_name: str = "gdn-all_ac-v0.0.2",
    evaluate_best: Annotated[
        bool, typer.Option(help="Evaluate the best model on the validation set after training.")
    ] = True,
):
    import torch
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeRemainingColumn,
    )
    from torch.utils.data import DataLoader

    import wandb
    from prc25.dataloader import VarlenDataset, collate_fn
    from prc25.datasets import preprocessed
    from prc25.model import FuelBurnPredictor, FuelBurnPredictorConfig

    train_dataset = VarlenDataset(partition=partition, split="train")
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dataset = VarlenDataset(partition=partition, split="validation")
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model_cfg = FuelBurnPredictorConfig(
        input_dim=len(preprocessed.TRAJECTORY_FEATURES),
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_aircraft_types=len(train_dataset.ac_type_vocab),
        aircraft_embedding_dim=aircraft_embedding_dim,
    )
    cfg = TrainConfig(
        partition=partition,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        model_config=model_cfg,
        project_name=project_name,
        exp_name=exp_name,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    exp_checkpoint_dir = PATH_CHECKPOINTS / cfg.exp_name
    exp_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    model = FuelBurnPredictor(cfg.model_config)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model architecture ({total_params=:,}):")
    for name, param in model.named_parameters():
        logger.info(f"  {name:<25} {tuple(param.shape)!s:<15} {param.numel():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss(reduction="none")

    wandb.init(project=cfg.project_name, name=cfg.exp_name, config=asdict(cfg))
    wandb.watch(model, log="all", log_freq=100)

    model.to(device)
    scaler = torch.amp.GradScaler(device=device, enabled=(device == "cuda"))
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
                x: torch.Tensor = data.x.to(device)
                y_log: torch.Tensor = data.y.to(device)
                offsets: torch.Tensor = data.offsets.to(device)
                aircraft_type_idx: torch.Tensor = data.aircraft_type_idx.to(device)
                durations: torch.Tensor = data.durations.to(device)

                optimizer.zero_grad()

                with torch.autocast(
                    device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")
                ):
                    y_pred_log = model(x, offsets, aircraft_type_idx)
                    loss_unweighted = criterion(y_pred_log, y_log)
                    loss = (loss_unweighted * durations.unsqueeze(1)).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if global_step > 0 and global_step % 25 == 0:
                    rmse_rate_val, rmse_kg_val, _ = _run_evaluation(
                        model, val_dataloader, device, progress, "validating"
                    )
                    wandb.log(
                        {"rmse_rate_val": rmse_rate_val, "rmse_kg_val": rmse_kg_val},
                        step=global_step,
                    )
                    if rmse_kg_val < best_val_rmse:
                        best_val_rmse = rmse_kg_val
                        checkpoint_path = exp_checkpoint_dir / "best.pt"
                        checkpoint = Checkpoint(
                            model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            scaler_state_dict=scaler.state_dict(),
                            global_step=global_step,
                            train_config=cfg,
                        )
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(
                            f"new best val rmse_rate: {rmse_rate_val:.4f} (rmse_kg: {rmse_kg_val:.2f}), saved to {checkpoint_path}"
                        )
                    model.train()

                batch_size = y_log.size(0)
                train_loss_total += loss.item() * batch_size
                train_samples += batch_size
                global_step += 1

                with torch.no_grad():
                    y_pred_rate_orig = torch.exp(y_pred_log.detach()) - 1.0
                    y_true_rate_orig = torch.exp(y_log.detach()) - 1.0
                    rmse_rate_orig = torch.sqrt(
                        torch.nn.functional.mse_loss(y_pred_rate_orig, y_true_rate_orig)
                    ).item()

                    y_pred_kg = y_pred_rate_orig * durations.unsqueeze(1)
                    y_true_kg = y_true_rate_orig * durations.unsqueeze(1)
                    rmse_kg = torch.sqrt(torch.nn.functional.mse_loss(y_pred_kg, y_true_kg)).item()
                wandb.log(
                    {"rmse_rate_train": rmse_rate_orig, "rmse_kg_train": rmse_kg},
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
        best_checkpoint_path = exp_checkpoint_dir / "best.pt"
        if best_checkpoint_path.exists():
            evaluate(best_checkpoint_path, partition, "validation", batch_size)
        else:
            logger.warning("no best model checkpoint found to evaluate.")


def _run_evaluation(
    model: FuelBurnPredictor,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    progress: Progress,
    description: str,
) -> tuple[float, float, pl.DataFrame]:
    import polars as pl
    import torch

    model.eval()
    all_preds, all_trues, all_segment_ids, all_durations = [], [], [], []

    with torch.no_grad():
        task = progress.add_task(description, total=len(dataloader))
        for data in dataloader:
            x, y_log, offsets, segment_ids, aircraft_type_idx, durations = (
                data.x.to(device),
                data.y.to(device),
                data.offsets.to(device),
                data.segment_ids.cpu(),
                data.aircraft_type_idx.to(device),
                data.durations.cpu(),
            )
            with torch.autocast(
                device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")
            ):
                y_pred_log = model(x, offsets, aircraft_type_idx)

            y_pred_orig = torch.exp(y_pred_log) - 1.0
            y_true_orig = torch.exp(y_log) - 1.0

            all_preds.append(y_pred_orig.cpu())
            all_trues.append(y_true_orig.cpu())
            all_segment_ids.append(segment_ids)
            all_durations.append(durations)
            progress.update(task, advance=1)
        progress.remove_task(task)

    preds_tensor = torch.cat(all_preds).flatten()
    trues_tensor = torch.cat(all_trues).flatten()
    segment_ids_tensor = torch.cat(all_segment_ids).flatten()
    durations_tensor = torch.cat(all_durations).flatten()

    df = pl.DataFrame(
        {
            "segment_id": segment_ids_tensor.numpy(),
            "duration_s": durations_tensor.numpy(),
            "y_true_rate": trues_tensor.numpy(),
            "y_pred_rate": preds_tensor.to(torch.float32).numpy(),
        }
    )
    df = df.with_columns(
        (pl.col("y_true_rate") * pl.col("duration_s")).alias("y_true_kg"),
        (pl.col("y_pred_rate") * pl.col("duration_s")).alias("y_pred_kg"),
    )

    rmse_rate = torch.sqrt(torch.mean((preds_tensor - trues_tensor) ** 2)).item()

    preds_kg = preds_tensor * durations_tensor
    trues_kg = trues_tensor * durations_tensor
    rmse_kg = torch.sqrt(torch.mean((preds_kg - trues_kg) ** 2)).item()

    return rmse_rate, rmse_kg, df


@app.command()
def evaluate(
    checkpoint_path: Path,
    partition: Partition = "phase1",
    split: Split = "validation",
    batch_size: int = 64,
):
    import torch
    from rich.progress import Progress
    from torch.utils.data import DataLoader

    from prc25.dataloader import VarlenDataset, collate_fn
    from prc25.model import FuelBurnPredictor, FuelBurnPredictorConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"using device: {device}")

    with torch.serialization.safe_globals([Checkpoint, TrainConfig, FuelBurnPredictorConfig]):
        checkpoint: Checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.train_config.model_config
    model = FuelBurnPredictor(model_config)
    model.load_state_dict(checkpoint.model_state_dict)
    model.to(device)

    dataset = VarlenDataset(partition=partition, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    with Progress() as progress:
        rmse_rate, rmse_kg, df = _run_evaluation(
            model, dataloader, device, progress, f"evaluating on {split} split"
        )

    exp_name = checkpoint_path.parent.name
    PATH_PREDICTIONS.mkdir(exist_ok=True, parents=True)
    output_path = PATH_PREDICTIONS / f"{exp_name}_{split}.parquet"
    df.write_parquet(output_path)
    logger.info(f"wrote predictions to {output_path}")
    logger.info(f"final rmse on {split=}: rate={rmse_rate:.4f} kg/s, total={rmse_kg:.2f} kg")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=False)]
    )
    app()
