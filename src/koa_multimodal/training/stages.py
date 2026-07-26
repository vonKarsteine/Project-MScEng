"""The stage registry: one place where a stage name resolves to everything else.

A stage name resolves to exactly one :class:`StageSpec` carrying every concern it
governs. Split across three axes instead -- the CLI stage name, a ``mode``
returned by the model builder that is only ever ``xray``/``mri``/``fusion``, and a
separately-passed ``loss_stage`` -- they drift apart silently, and branches keyed
on the wrong axis become unreachable: no ``mode`` is ever ``"qat"``, so a QAT arm
of a batch-size helper is dead code and any config key feeding it is never read.
One spec means they cannot drift and an unreachable branch cannot exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple

from koa_multimodal.config.schema import Config
from koa_multimodal.core.errors import ConfigError
from koa_multimodal.core.ids import FUSION_ROUTES


class Modality(str, Enum):
    """What the stage feeds the model. Drives batching and whether MRI loads."""

    XRAY = "xray"
    MRI = "mri"
    FUSION = "fusion"


class Objective(str, Enum):
    """Which loss the stage optimises."""

    ORDINAL = "ordinal"
    SIMSIAM = "simsiam"
    CONTRASTIVE = "contrastive"
    RCKF = "rckf"
    QAT = "qat"


@dataclass(frozen=True)
class StageSpec:
    """A stage name and everything it determines."""

    name: str
    modality: Modality
    objective: Objective
    description: str
    #: QAT trains a student against a frozen teacher rather than from scratch.
    distills: bool = False

    def batch_size(self, config: Config) -> int:
        training = config.training
        if self.modality is Modality.XRAY:
            return training.xray_batch_size
        if self.modality is Modality.MRI:
            return training.mri_batch_size
        return training.fusion_batch_size

    def gradient_accumulation(self, config: Config) -> int:
        """Accumulation applies to the two volumetric paths.

        Table 4.3 reports a global batch size of 16. MRI decode and 3-D resize
        cost forces a physical ``mri_batch_size`` of 1, so the reported size is
        reached by accumulation rather than by a larger physical batch.
        """

        if self.modality is Modality.XRAY:
            return 1
        return config.training.multimodal_gradient_accumulation

    def effective_batch_size(self, config: Config) -> int:
        return self.batch_size(config) * self.gradient_accumulation(config)

    @property
    def loads_mri(self) -> bool:
        return self.modality in (Modality.MRI, Modality.FUSION)

    def describe(self, config: Optional[Config] = None) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "stage": self.name,
            "modality": self.modality.value,
            "objective": self.objective.value,
            "loadsMri": self.loads_mri,
            "distills": self.distills,
            "description": self.description,
        }
        if config is not None:
            payload.update(
                {
                    "batchSize": self.batch_size(config),
                    "gradientAccumulation": self.gradient_accumulation(config),
                    "effectiveBatchSize": self.effective_batch_size(config),
                }
            )
        return payload


def _fusion_stage(route: str, objective: Objective, description: str) -> StageSpec:
    return StageSpec(
        name=route, modality=Modality.FUSION, objective=objective, description=description
    )


STAGES: Mapping[str, StageSpec] = {
    "xray": StageSpec(
        name="xray",
        modality=Modality.XRAY,
        objective=Objective.ORDINAL,
        description="Train one X-ray unimodal candidate (Table 4.5).",
    ),
    "mri": StageSpec(
        name="mri",
        modality=Modality.MRI,
        objective=Objective.ORDINAL,
        description="Train one MRI unimodal candidate (Table 4.6).",
    ),
    "mri_ssl": StageSpec(
        name="mri_ssl",
        modality=Modality.MRI,
        objective=Objective.SIMSIAM,
        description=(
            "SimSiam self-supervised pretraining of the MRI encoder. KL labels are "
            "native to radiographs, so supervising MRI with them directly imports "
            "label misalignment; SSL learns the volumetric representation first."
        ),
    ),
    "late_concat": _fusion_stage(
        "late_concat", Objective.ORDINAL, "Concatenation fusion baseline (Table 4.7)."
    ),
    "gated": _fusion_stage(
        "gated", Objective.ORDINAL, "Gated multimodal unit fusion (Table 4.7)."
    ),
    "cross_attention": _fusion_stage(
        "cross_attention", Objective.ORDINAL, "Cross-attention fusion (Table 4.7)."
    ),
    "contrastive": _fusion_stage(
        "contrastive",
        Objective.CONTRASTIVE,
        "CLIP-style contrastive fusion, CORN + lambda_align * InfoNCE (Table 4.8).",
    ),
    "rckf": _fusion_stage(
        "rckf",
        Objective.RCKF,
        "Residual-Calibrated Kalman Fusion, CORN + alpha_res * innovation NLL (Table 4.9).",
    ),
    "qat": StageSpec(
        name="qat",
        modality=Modality.FUSION,
        objective=Objective.QAT,
        description=(
            "Distil the frozen FP32 composite teacher into a quantized student "
            "under the six-term deployment objective (Table 4.15)."
        ),
        distills=True,
    ),
}

STAGE_NAMES: Tuple[str, ...] = tuple(STAGES)
FUSION_STAGE_NAMES: Tuple[str, ...] = tuple(FUSION_ROUTES)


def get_stage(name: str) -> StageSpec:
    try:
        return STAGES[name]
    except KeyError:
        raise ConfigError(
            f"Unknown stage {name!r}. Available stages: {', '.join(STAGE_NAMES)}"
        ) from None
