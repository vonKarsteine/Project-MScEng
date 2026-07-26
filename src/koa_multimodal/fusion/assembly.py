"""Assemble a fusion route from two trained branch encoders.

Branch checkpoints **warm-start the encoders only**: both branch CORN heads are
discarded, because the fusion route owns a new head over a new latent space and a
transplanted head would be scored on features it never saw.

The graft relies on an attribute asymmetry that is load-bearing --
``XrayCandidate`` stores its network as ``.backbone`` while MRI candidates store
theirs as ``.encoder`` -- so the prefix to strip is known statically per modality.
The load is then **strict**: a checkpoint that supplies only a handful of
coincidentally-named tensors is rejected, because the permissive alternative
produces a mostly-random encoder that reports success.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

from torch import nn

from koa_multimodal.core.checkpoint import (
    StateLoadReport,
    load_checkpoint,
    load_module_state,
    strip_prefix,
)
from koa_multimodal.core.errors import KoaError
from koa_multimodal.core.ids import FUSION_ROUTES, fusion_candidate_id
from koa_multimodal.config.schema import ModelConfig, RckfConfig
from koa_multimodal.fusion.base import FusionBase
from koa_multimodal.fusion.contrastive import ContrastiveFusion
from koa_multimodal.fusion.rckf import RCKFFusion
from koa_multimodal.fusion.traditional import (
    CrossAttentionFusion,
    GatedFusion,
    LateConcatFusion,
)

PathLike = Union[str, Path]

#: Route id -> constructor. Keys match ``koa_multimodal.core.ids.FUSION_ROUTES``.
FUSION_BUILDERS = {
    "late_concat": LateConcatFusion,
    "gated": GatedFusion,
    "cross_attention": CrossAttentionFusion,
    "contrastive": ContrastiveFusion,
    "rckf": RCKFFusion,
}

#: The attribute each candidate type stores its network under, and hence the
#: state-dict prefix to strip when grafting.
BRANCH_ATTRIBUTES = {"xray": "backbone", "mri": "encoder"}


def build_fusion(
    route: str,
    xray_candidate_id: str,
    mri_candidate_id: str,
    *,
    model_cfg: Optional[ModelConfig] = None,
    rckf_cfg: Optional[RckfConfig] = None,
    pretrained: Optional[bool] = None,
) -> FusionBase:
    """Construct one fusion route over two freshly built branch encoders."""

    from koa_multimodal.models.catalog import build_base_candidate

    if route not in FUSION_BUILDERS:
        raise KoaError(f"Unknown fusion route {route!r}; expected one of {FUSION_ROUTES}")
    model_cfg = model_cfg or ModelConfig()

    _, xray_candidate = build_base_candidate(
        xray_candidate_id, model_cfg=model_cfg, pretrained=pretrained
    )
    _, mri_candidate = build_base_candidate(
        mri_candidate_id, model_cfg=model_cfg, pretrained=pretrained
    )

    kwargs: Dict[str, Any] = {
        "feature_dim": model_cfg.feature_dim,
        "num_classes": model_cfg.num_classes,
        "candidate_id": fusion_candidate_id(route, xray_candidate_id, mri_candidate_id),
        # The branch CORN heads are deliberately left behind here.
        "xray_encoder": getattr(xray_candidate, BRANCH_ATTRIBUTES["xray"]),
        "mri_encoder": getattr(mri_candidate, BRANCH_ATTRIBUTES["mri"]),
    }
    if route == "rckf":
        rckf_cfg = rckf_cfg or RckfConfig()
        kwargs.update(
            prior_variance=rckf_cfg.prior_variance,
            variance_floor=rckf_cfg.variance_floor,
            variance_ceiling=rckf_cfg.variance_ceiling,
            variance_groups=rckf_cfg.variance_groups,
            activation=rckf_cfg.monotone_activation,
        )
    return FUSION_BUILDERS[route](**kwargs)


def load_branch_encoder(
    encoder: nn.Module,
    checkpoint_path: PathLike,
    modality: str,
) -> StateLoadReport:
    """Warm-start one branch encoder from a candidate checkpoint.

    Strips the modality's known prefix and loads strictly. Refusing a partial load
    is the point: a mismatched checkpoint that quietly populates 1 % of the
    tensors yields a near-random encoder that reports success and produces
    plausible-looking metrics.
    """

    if modality not in BRANCH_ATTRIBUTES:
        raise KoaError(f"Unknown modality {modality!r}; expected 'xray' or 'mri'")
    payload = load_checkpoint(checkpoint_path, trusted=True)
    state = payload.get("model_state_dict") or payload.get("state_dict") or payload

    prefix = BRANCH_ATTRIBUTES[modality]
    branch_state = strip_prefix(state, prefix)
    if not branch_state:
        raise KoaError(
            f"No tensors under the {prefix!r} prefix in {checkpoint_path}. This does not "
            f"look like a {modality} candidate checkpoint; refusing to guess."
        )
    return load_module_state(encoder, branch_state, strict=True)


def assemble_from_checkpoints(
    route: str,
    xray_candidate_id: str,
    mri_candidate_id: str,
    *,
    xray_checkpoint: Optional[PathLike] = None,
    mri_checkpoint: Optional[PathLike] = None,
    model_cfg: Optional[ModelConfig] = None,
    rckf_cfg: Optional[RckfConfig] = None,
    pretrained: Optional[bool] = None,
) -> "AssembledFusion":
    """Build a route and warm-start whichever branches have a checkpoint."""

    model = build_fusion(
        route,
        xray_candidate_id,
        mri_candidate_id,
        model_cfg=model_cfg,
        rckf_cfg=rckf_cfg,
        pretrained=pretrained,
    )
    reports: Dict[str, StateLoadReport] = {}
    if xray_checkpoint is not None:
        reports["xray"] = load_branch_encoder(model.xray_encoder, xray_checkpoint, "xray")
    if mri_checkpoint is not None:
        reports["mri"] = load_branch_encoder(model.mri_encoder, mri_checkpoint, "mri")
    return AssembledFusion(model=model, reports=reports)


class AssembledFusion:
    """A fusion route plus an auditable record of what its warm start actually loaded."""

    def __init__(self, model: FusionBase, reports: Dict[str, StateLoadReport]) -> None:
        self.model = model
        self.reports = reports

    def to_payload(self) -> Dict[str, Any]:
        return {
            "candidateId": self.model.candidate_id,
            "fusionType": self.model.fusion_type,
            "featureDim": self.model.feature_dim,
            "branchWarmStart": {
                modality: {
                    "matched": report.matched,
                    "missing": report.missing,
                    "unexpected": report.unexpected,
                    "totalTarget": report.total_target,
                }
                for modality, report in self.reports.items()
            },
        }
