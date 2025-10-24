from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet  # see: https://arxiv.org/pdf/2412.06464

logger = logging.getLogger(__name__)


class ZeroCentredRMSNorm(nn.Module):
    """Avoids abnormal amplification of some weights in the original QK-norm.
    During regularisation and weight decay, `weight` will be pushed near 0.

    See: https://ceramic.ai/blog/zerocentered"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (hidden_states * (1.0 + self.weight)).to(input_dtype)


class Pooler(nn.Module):
    def __init__(self, mode: Literal["mean", "last"] = "last"):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        if self.mode == "last":
            return x[cu_seqlens[1:] - 1]
        elif self.mode == "mean":
            indices = torch.arange(x.size(0), device=x.device)
            return torch.nn.functional.embedding_bag(
                indices, x, offsets=cu_seqlens[:-1], mode="mean"
            )
        else:
            raise ValueError(f"unknown {self.mode=}")


@dataclass
class FuelBurnPredictorConfig:
    input_dim: int
    hidden_size: int
    num_heads: int
    num_aircraft_types: int
    aircraft_embedding_dim: int


class FuelBurnPredictor(nn.Module):
    """Terminology:

    We have three *segments* of the trajectory, which range from 2 to thousands of tokens:
        - [takeoff, start]
        - [start, end]: the segment for which we predict fuel burn
        - [end, arrival]

    This model only processes the [start, end] segment.

    Instead of padding, segments are tightly packed together in a long tensor, and
    FLA is informed of segment boundaries via the `cu_seqlens` tensor.
    """

    def __init__(self, cfg: FuelBurnPredictorConfig):
        super().__init__()
        from .hacks import fla_autotuner_check_removed_nb

        fla_autotuner_check_removed_nb()

        self.config = cfg
        key_dim = int(cfg.hidden_size * 0.75)  # as per docstring
        assert key_dim % cfg.num_heads == 0, (
            "int(hidden_size * 0.75) must be divisible by num_heads (use_gate=True)"
        )
        head_dim = key_dim // cfg.num_heads

        self.aircraft_embedding = nn.Embedding(cfg.num_aircraft_types, cfg.aircraft_embedding_dim)
        self.input_proj = nn.Linear(cfg.input_dim + cfg.aircraft_embedding_dim, cfg.hidden_size)
        self.norm = ZeroCentredRMSNorm(cfg.hidden_size)
        self.gdn = GatedDeltaNet(
            hidden_size=cfg.hidden_size, num_heads=cfg.num_heads, head_dim=head_dim
        )
        self.pooler = Pooler()
        self.regression_head = nn.Linear(cfg.hidden_size, 1)

    def forward(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor, aircraft_type_idx: torch.Tensor
    ) -> torch.Tensor:
        """:param x: Packed tensor
        :param cu_seqlens: Cumulative sequence lengths for packed tensor"""
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        ac_embeddings = self.aircraft_embedding(aircraft_type_idx)
        ac_embeddings_repeated = torch.repeat_interleave(ac_embeddings, lengths, dim=0)

        x = torch.cat([x, ac_embeddings_repeated], dim=1)
        x = self.input_proj(x)

        residual = x
        x = self.norm(x)
        x, _, _ = self.gdn(x.unsqueeze(0), cu_seqlens=cu_seqlens)
        x = x.squeeze(0)
        x = x + residual

        pooled_x = self.pooler(x, cu_seqlens)
        y_pred = self.regression_head(pooled_x)
        return y_pred
