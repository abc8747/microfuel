from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet

logger = logging.getLogger(__name__)


class ZeroCentredRMSNorm(nn.Module):
    """Avoids abnormal amplification of some weights in the original QK-norm.
    During regularization and weight decay, `weight` will be pushed near 0.
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


class VarlenPooler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        pooled_outputs = []
        for i in range(len(offsets) - 1):
            start, end = offsets[i], offsets[i + 1]
            segment = x[start:end]
            pooled = segment[-1]
            pooled_outputs.append(pooled)
        return torch.stack(pooled_outputs)


@dataclass
class FuelBurnPredictorConfig:
    input_dim: int
    hidden_size: int
    num_heads: int


class FuelBurnPredictor(nn.Module):
    """Gated Delta Network: https://arxiv.org/pdf/2412.06464"""

    def __init__(self, cfg: FuelBurnPredictorConfig):
        super().__init__()
        self.config = cfg
        key_dim = int(cfg.hidden_size * 0.75)  # as per docstring
        assert key_dim % cfg.num_heads == 0, (
            "int(hidden_size * 0.75) must be divisible by num_heads (use_gate=True)"
        )
        head_dim = key_dim // cfg.num_heads

        self.input_proj = nn.Linear(cfg.input_dim, cfg.hidden_size)
        self.norm = ZeroCentredRMSNorm(cfg.hidden_size)
        self.gdn = GatedDeltaNet(
            hidden_size=cfg.hidden_size, num_heads=cfg.num_heads, head_dim=head_dim
        )  # NOTE: gdn.o_norm.weight is not zero-centred so it must not be weight-decayed!
        self.pooler = VarlenPooler()
        self.regression_head = nn.Linear(cfg.hidden_size, 1)

    def forward(self, x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.norm(x)
        x, _, _ = self.gdn(x.unsqueeze(0), cu_seqlens=offsets)
        pooled_x = self.pooler(x.squeeze(0), offsets)
        y_pred = self.regression_head(pooled_x)
        return y_pred
