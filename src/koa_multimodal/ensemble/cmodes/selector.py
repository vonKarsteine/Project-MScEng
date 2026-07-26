"""The shared scalar risk-reduction regressor.

One small MLP scores **every** route with the **same** weights, taking the 16-D
pairwise vector for a (sample, route) pair and returning a scalar estimate of the
risk reduction that switching to that route would buy.

Sharing the weights is what makes the selector permutation-equivariant over the
candidate pool: reordering the pool reorders the scores and changes nothing else.
A per-route head would instead let the selector memorise "route 3 is usually good",
which is precisely the static-ensemble failure mode C-MODES exists to avoid.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from koa_multimodal.core.errors import ContractError


class CMODESSelectorModel(nn.Module):
    """``g_theta(a_i, b_ik) -> scalar`` shared across all routes."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``[..., input_dim]`` -> ``[...]``. The trailing scalar axis is squeezed."""

        if features.shape[-1] != self.input_dim:
            raise ContractError(
                f"Selector expects feature dim {self.input_dim}, got {features.shape[-1]}. "
                "The 16-D layout is recorded in the checkpoint's featureSpec; a "
                "mismatch means the pairwise builder and the checkpoint disagree."
            )
        return self.net(features).squeeze(-1)


def build_selector(
    input_dim: int = 16,
    hidden_dim: int = 64,
    device: Optional[torch.device] = None,
) -> CMODESSelectorModel:
    model = CMODESSelectorModel(input_dim=input_dim, hidden_dim=hidden_dim)
    return model.to(device) if device is not None else model
