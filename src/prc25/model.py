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


class LinearAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.norm = ZeroCentredRMSNorm(hidden_size)
        self.gdn = GatedDeltaNet(hidden_size=hidden_size, num_heads=num_heads, head_dim=head_dim)

    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, _, _ = self.gdn(x.unsqueeze(0), cu_seqlens=cu_seqlens)
        x = x.squeeze(0)
        x = x + residual
        return x


class StaticHyperNet(nn.Module):
    """Creates a specialised feature extractor for each aircraft type, improving over
    feature conditioning (concatenating embeddings to input).

    See: https://arxiv.org/pdf/1609.09106#page=3 (Section 3.1)."""

    def __init__(
        self, num_aircraft_types: int, embedding_dim: int, input_dim: int, output_dim: int
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(num_aircraft_types, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, (input_dim * output_dim) + output_dim),
        )

    def forward(
        self, aircraft_type_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(aircraft_type_idx)
        params = self.mlp(embeddings)
        weights_flat = params[:, : self.input_dim * self.output_dim]
        bias = params[:, self.input_dim * self.output_dim :]
        weights = weights_flat.view(-1, self.output_dim, self.input_dim)
        return weights, bias, embeddings


@dataclass
class FuelBurnPredictorConfig:
    input_dim: int
    hidden_size: int
    num_heads: int
    num_aircraft_types: int
    aircraft_embedding_dim: int
    num_layers: int
    pooler_mode: Literal["mean", "last"]


class FuelBurnPredictor(nn.Module):
    """Terminology:

    We have three *segments* of the trajectory, which range from 2 to thousands of tokens:
        - [takeoff, start]
        - [start, end]: the segment for which we predict fuel burn
        - [end, arrival]

    This model processes the [start, end] segment and the full [takeoff, arrival] flight.

    Instead of padding, sequences are tightly packed together in a long tensor, and
    FLA is informed of boundaries via the `cu_seqlens` tensor.
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

        # segment processing branch
        self.hypernetwork_segment = StaticHyperNet(
            num_aircraft_types=cfg.num_aircraft_types,
            embedding_dim=cfg.aircraft_embedding_dim,
            input_dim=cfg.input_dim,
            output_dim=cfg.hidden_size,
        )
        self.layers_segment = nn.ModuleList(
            [
                LinearAttentionBlock(cfg.hidden_size, cfg.num_heads, head_dim)
                for _ in range(cfg.num_layers)
            ]
        )
        self.pooler_segment = Pooler(mode=cfg.pooler_mode)

        # flight context processing branch
        self.hypernetwork_flight = StaticHyperNet(
            num_aircraft_types=cfg.num_aircraft_types,
            embedding_dim=cfg.aircraft_embedding_dim,
            input_dim=cfg.input_dim,
            output_dim=cfg.hidden_size,
        )
        self.layers_flight = nn.ModuleList(
            [
                LinearAttentionBlock(cfg.hidden_size, cfg.num_heads, head_dim)
                for _ in range(cfg.num_layers)
            ]
        )
        self.pooler_flight = Pooler(mode=cfg.pooler_mode)

        # share embedding layer between hypernetworks
        self.hypernetwork_flight.embedding = self.hypernetwork_segment.embedding

        self.regression_head = nn.Linear(
            cfg.hidden_size + cfg.hidden_size + cfg.aircraft_embedding_dim, 1
        )

    def forward(
        self,
        x_flight: torch.Tensor,
        cu_seqlens_flight: torch.Tensor,
        x_segment: torch.Tensor,
        cu_seqlens_segment: torch.Tensor,
        aircraft_type_idx: torch.Tensor,
    ) -> torch.Tensor:
        """:param x_flight: packed tensor of full flight trajectories
        :param cu_seqlens_flight: cumulative sequence lengths for flight tensor
        :param x_segment: packed tensor of trajectory segments for prediction
        :param cu_seqlens_segment: cumulative sequence lengths for segment tensor
        :param aircraft_type_idx: (B,) tensor of aircraft type indices"""
        # segment processing
        segment_lengths = cu_seqlens_segment[1:] - cu_seqlens_segment[:-1]
        weights_s, bias_s, ac_embeddings = self.hypernetwork_segment(aircraft_type_idx)
        weights_expanded_s = torch.repeat_interleave(weights_s, segment_lengths, dim=0)
        bias_expanded_s = torch.repeat_interleave(bias_s, segment_lengths, dim=0)
        x_s = torch.bmm(weights_expanded_s, x_segment.unsqueeze(-1)).squeeze(-1) + bias_expanded_s

        for layer in self.layers_segment:
            x_s = layer(x_s, cu_seqlens_segment)
        pooled_segment = self.pooler_segment(x_s, cu_seqlens_segment)

        # flight context processing
        flight_lengths = cu_seqlens_flight[1:] - cu_seqlens_flight[:-1]
        weights_f, bias_f, _ = self.hypernetwork_flight(aircraft_type_idx)
        weights_expanded_f = torch.repeat_interleave(weights_f, flight_lengths, dim=0)
        bias_expanded_f = torch.repeat_interleave(bias_f, flight_lengths, dim=0)
        x_f = torch.bmm(weights_expanded_f, x_flight.unsqueeze(-1)).squeeze(-1) + bias_expanded_f

        for layer in self.layers_flight:
            x_f = layer(x_f, cu_seqlens_flight)
        pooled_flight = self.pooler_flight(x_f, cu_seqlens_flight)

        # final regression
        combined_features = torch.cat([pooled_segment, pooled_flight, ac_embeddings], dim=1)
        y_pred = self.regression_head(combined_features)
        return y_pred
