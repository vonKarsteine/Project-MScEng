"""Simulated federated aggregation of the selector -- §3.3.3.

Each candidate route is treated as a **pseudo-client** ``c_k`` holding its own
out-of-fold risk-differential dataset ``D_k`` (eq. 3.21). The differentials are
strongly non-IID across routes -- a route's data reflects its own expertise -- so
plain averaging lets a high-variance client dominate the shared selector. The
server update is FedYogi (Reddi et al., 2021), whose adaptive second moment damps
exactly that.

**The sign convention is the thing to get right.** The thesis defines the client
delta as ``Delta_k = theta_global - theta_local`` (eq. 3.23) -- global *minus*
local, the opposite of the usual local-minus-global -- and the server then
**subtracts** the scaled step (eq. 3.25):

    m_t = beta_1 m_{t-1} + (1 - beta_1) g_t
    v_t = v_{t-1} - (1 - beta_2) sign(v_{t-1} - g_t^2) g_t^2
    theta_{t+1} = theta_t - alpha m_t / (sqrt(v_t) + tau)

Flipping either sign passes every shape check and silently ascends the loss, so
``tests/test_cmodes.py`` asserts the convention directly rather than trusting it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.nn import functional as F

from koa_multimodal.core.errors import KoaError
from koa_multimodal.ensemble.cmodes.risk import RiskDifferentialDataset
from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel


@dataclass
class FedYogiState:
    """Server-side moment estimates, persisted with the selector checkpoint."""

    first_moment: Dict[str, torch.Tensor] = field(default_factory=dict)
    second_moment: Dict[str, torch.Tensor] = field(default_factory=dict)
    step: int = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "firstMoment": {k: v.detach().cpu() for k, v in self.first_moment.items()},
            "secondMoment": {k: v.detach().cpu() for k, v in self.second_moment.items()},
            "step": int(self.step),
        }

    @classmethod
    def from_state_dict(cls, payload: Dict[str, Any]) -> "FedYogiState":
        return cls(
            first_moment={
                k: v.detach().clone() for k, v in payload.get("firstMoment", {}).items()
            },
            second_moment={
                k: v.detach().clone() for k, v in payload.get("secondMoment", {}).items()
            },
            step=int(payload.get("step", 0)),
        )


def local_update(
    global_model: CMODESSelectorModel,
    dataset: RiskDifferentialDataset,
    candidate_index: int,
    *,
    learning_rate: float = 1e-3,
    local_steps: int = 3,
) -> Dict[str, torch.Tensor]:
    """Run one pseudo-client's local descent and return ``theta_global - theta_local``.

    Smooth L1 rather than MSE: risk differentials have heavy tails -- a candidate
    that is confidently wrong on one sample produces a very large ``-log p`` -- and
    a squared loss would let those few samples dictate the selector.
    """

    model = deepcopy(global_model)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    features, targets = dataset.client_data(candidate_index)
    device = next(model.parameters()).device
    features, targets = features.to(device), targets.to(device)

    for _ in range(max(1, int(local_steps))):
        optimizer.zero_grad(set_to_none=True)
        F.smooth_l1_loss(model(features), targets).backward()
        optimizer.step()

    global_parameters = dict(global_model.named_parameters())
    local_parameters = dict(model.named_parameters())
    # eq. 3.23: global MINUS local. Not the usual local-minus-global.
    return {
        name: global_parameters[name].detach() - local_parameters[name].detach()
        for name in global_parameters
    }


def fedyogi_aggregate(
    model: CMODESSelectorModel,
    updates: Sequence[Dict[str, torch.Tensor]],
    weights: Sequence[float],
    state: Optional[FedYogiState] = None,
    *,
    server_learning_rate: float = 0.5,
    beta1: float = 0.9,
    beta2: float = 0.999,
    tau: float = 1e-3,
) -> FedYogiState:
    """Apply one FedYogi server step in place. Eqs. 3.24-3.25."""

    if not updates:
        raise KoaError("At least one pseudo-client update is required")
    if len(updates) != len(weights):
        raise KoaError("FedYogi update count and weight count must match")
    values = [float(value) for value in weights]
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise KoaError("FedYogi weights must be non-negative with positive total mass")

    total = sum(values)
    normalized = [value / total for value in values]
    state = state or FedYogiState()

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            # eq. 3.24: the weighted pseudo-gradient.
            aggregated = sum(
                update[name].to(parameter.device) * weight
                for update, weight in zip(updates, normalized)
            )
            first = state.first_moment.get(name, torch.zeros_like(parameter)).to(parameter.device)
            second = state.second_moment.get(name, torch.zeros_like(parameter)).to(parameter.device)

            first = beta1 * first + (1.0 - beta1) * aggregated
            squared = aggregated.square()
            # Yogi's controlled second moment: it can only decrease by a bounded
            # amount per step, so a single large client cannot spike the
            # denominator and stall every subsequent update.
            second = (second - (1.0 - beta2) * torch.sign(second - squared) * squared).clamp_min(0.0)

            # eq. 3.25: SUBTRACT the scaled step.
            parameter.sub_(server_learning_rate * first / (second.sqrt() + tau))

            state.first_moment[name] = first.detach()
            state.second_moment[name] = second.detach()

    state.step += 1
    return state


def train_selector(
    model: CMODESSelectorModel,
    dataset: RiskDifferentialDataset,
    *,
    rounds: int = 5,
    local_steps: int = 3,
    local_learning_rate: float = 1e-3,
    server_learning_rate: float = 0.5,
    beta1: float = 0.9,
    beta2: float = 0.999,
    tau: float = 1e-3,
) -> FedYogiState:
    """The full simulated federated loop over the candidate pseudo-clients."""

    state = FedYogiState()
    # Every route holds the same number of rows, so the data weights w_k of
    # eq. 3.24 are uniform here; the term is kept explicit because it is what the
    # methodology specifies and what a differently-sized pool would need.
    weights = [1.0] * dataset.candidate_count
    for _ in range(max(1, int(rounds))):
        updates: List[Dict[str, torch.Tensor]] = [
            local_update(
                model,
                dataset,
                index,
                learning_rate=local_learning_rate,
                local_steps=local_steps,
            )
            for index in range(dataset.candidate_count)
        ]
        state = fedyogi_aggregate(
            model,
            updates,
            weights,
            state,
            server_learning_rate=server_learning_rate,
            beta1=beta1,
            beta2=beta2,
            tau=tau,
        )
    return state
