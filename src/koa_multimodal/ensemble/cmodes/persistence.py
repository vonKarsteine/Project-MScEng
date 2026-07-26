"""Persist and restore a fitted C-MODES selector.

The checkpoint carries everything needed to rebuild the deployed router: the
selector weights, the FedYogi server state, the candidate pool and its order, the
16-D feature spec (so a reader can verify the pairwise layout the weights expect),
and the OOF-calibrated ``tau_s`` / ``c_switch``.

The calibrated threshold lives **here** rather than in ``configs/default.toml``
because it is an output of calibration, not an input to it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch

from koa_multimodal.core.checkpoint import load_checkpoint, save_checkpoint
from koa_multimodal.core.errors import KoaError
from koa_multimodal.ensemble.cmodes.calibration import SwitchCalibration
from koa_multimodal.ensemble.cmodes.features import RiskFeatureBuilder
from koa_multimodal.ensemble.cmodes.federated import FedYogiState
from koa_multimodal.ensemble.cmodes.routing import CMODESRouter
from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel

PathLike = Union[str, Path]

CMODES_CHECKPOINT_SCHEMA = "cmodes-selector-v2"


def save_selector(
    path: PathLike,
    model: CMODESSelectorModel,
    state: FedYogiState,
    *,
    candidate_ids: Sequence[str],
    default_candidate_id: str,
    feature_builder: RiskFeatureBuilder,
    calibration: Optional[SwitchCalibration] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "schemaVersion": CMODES_CHECKPOINT_SCHEMA,
        "modelStateDict": model.state_dict(),
        "fedYogiState": state.state_dict(),
        "candidateIds": list(candidate_ids),
        "defaultCandidateId": str(default_candidate_id),
        "featureSpec": feature_builder.feature_spec(),
        "modelConfig": {"inputDim": model.input_dim, "hiddenDim": model.hidden_dim},
        "hyperparameters": dict(hyperparameters or {}),
    }
    if calibration is not None:
        payload["calibration"] = calibration.to_payload()
        payload["hyperparameters"].update(
            {"tauS": calibration.tau_s, "switchCost": calibration.c_switch}
        )
    return save_checkpoint(payload, path)


def load_selector(
    path: PathLike, *, map_location: Union[str, torch.device] = "cpu"
) -> Tuple[CMODESSelectorModel, FedYogiState, Dict[str, Any]]:
    payload = load_checkpoint(path, map_location=map_location, trusted=True)
    schema = payload.get("schemaVersion")
    if schema != CMODES_CHECKPOINT_SCHEMA:
        raise KoaError(
            f"Unsupported C-MODES checkpoint schema {schema!r}, expected "
            f"{CMODES_CHECKPOINT_SCHEMA!r}"
        )
    config = payload["modelConfig"]
    model = CMODESSelectorModel(
        input_dim=int(config["inputDim"]), hidden_dim=int(config["hiddenDim"])
    )
    model.load_state_dict(payload["modelStateDict"])
    state = FedYogiState.from_state_dict(payload["fedYogiState"])
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in ("modelStateDict", "fedYogiState")
    }
    return model, state, metadata


def router_from_checkpoint(
    path: PathLike,
    *,
    tau_s: Optional[float] = None,
    c_switch: Optional[float] = None,
) -> CMODESRouter:
    """Rebuild the deployed router, threshold included.

    Explicit ``tau_s`` / ``c_switch`` override the calibrated pair; otherwise the
    checkpoint's own values are used. Only their sum affects the rule.
    """

    model, _state, metadata = load_selector(path)
    candidate_ids = [str(value) for value in metadata.get("candidateIds", [])]
    if not candidate_ids:
        raise KoaError("C-MODES checkpoint does not record a candidate pool")

    builder = RiskFeatureBuilder.from_spec(metadata.get("featureSpec"))
    if builder.input_dim != model.input_dim:
        raise KoaError(
            f"Checkpoint featureSpec declares inputDim {builder.input_dim} but the "
            f"selector expects {model.input_dim}; the pairwise layout and the "
            "weights disagree."
        )

    hyperparameters = metadata.get("hyperparameters", {})
    return CMODESRouter(
        selector_model=model,
        candidate_ids=candidate_ids,
        default_candidate_id=str(metadata.get("defaultCandidateId", candidate_ids[0])),
        tau_s=float(tau_s if tau_s is not None else hyperparameters.get("tauS", 0.0)),
        c_switch=float(
            c_switch if c_switch is not None else hyperparameters.get("switchCost", 0.04)
        ),
        feature_builder=builder,
    )
