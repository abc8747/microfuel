from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from rich.logging import RichHandler

import microfuel
import microfuel.datasets
import microfuel.model
from microfuel.model import FuelBurnPredictorConfig, TrainConfig, TrainingState

sys.modules["prc25"] = microfuel
sys.modules["prc25.model"] = microfuel.model
sys.modules["prc25.datasets"] = microfuel.datasets

logger = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True)


# required for v0.0.0 checkpoint loading
@dataclass
class Checkpoint:
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scaler_state_dict: dict[str, Any]
    global_step: int
    train_config: TrainConfig


@app.command()
def migrate(
    input_fp: Annotated[
        Path, typer.Argument(help="Path to old monolithic .pt file", exists=True, dir_okay=False)
    ],
    output_fp: Annotated[
        Path, typer.Argument(help="Path for the new model weights .pt file", dir_okay=False)
    ],
):  # 0.0.0 -> 0.0.1  ONLY
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    assert output_fp.stem.endswith("_model"), "output filename must end with '_model'"
    name = output_fp.stem.removesuffix("_model")
    state_path = output_fp.with_name(f"{name}_state.pt")
    config_path = output_fp.parent / "config.json"

    with torch.serialization.safe_globals([Checkpoint, TrainConfig, FuelBurnPredictorConfig]):
        ckpt = torch.load(input_fp, map_location="cpu", weights_only=False)

    logger.info(f"writing config to {config_path}")
    with open(config_path, "w") as f:
        train_config: TrainConfig = ckpt.train_config
        json.dump(asdict(train_config), f, indent=2)

    logger.info(f"writing model weights to {output_fp}")
    torch.save(ckpt.model_state_dict, output_fp)

    logger.info(f"writing training state to {state_path}")
    state = TrainingState(
        optimizer_state_dict=ckpt.optimizer_state_dict,
        scaler_state_dict=ckpt.scaler_state_dict,
        global_step=ckpt.global_step,
    )
    torch.save(state, state_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=False)]
    )
    app()
