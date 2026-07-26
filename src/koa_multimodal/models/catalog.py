"""The base-candidate registry: nine specifications and one builder.

These nine ids are primary keys, not labels. They name checkpoint directories,
key the per-candidate rows of every prediction artifact, index the candidate axis
of the C-MODES route table, and appear verbatim inside the long fusion ids
composed by :func:`koa_multimodal.core.ids.fusion_candidate_id`. Renaming one
does not rename its history, so a change here orphans every artifact that
mentions the id it replaced.

The tuple order is the order of Tables 4.5 and 4.6, best first. It is
therefore reportable rather than arbitrary, and anything that iterates the
catalogue to build a table gets the published ordering for free.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

from torch import nn

from koa_multimodal.config.schema import ModelConfig
from koa_multimodal.core.errors import KoaError
from koa_multimodal.models.candidates import MriCandidate, SimSiamMriCandidate, XrayCandidate
from koa_multimodal.models.mri import (
    R3D18AttentionMriBackbone,
    R3D18MriBackbone,
    Swin3DMriBackbone,
)
from koa_multimodal.models.xray import TimmXrayBackbone


@dataclass(frozen=True)
class CandidateSpec:
    """Everything about a base candidate that is true before it is built.

    ``builder_key`` selects the construction branch in
    :func:`build_base_candidate`; ``backbone_name`` is the architecture
    identifier, and matches the ``backbone`` diagnostic the built module reports
    on every prediction, so a record can be traced back to this row.
    """

    candidate_id: str
    modality: str
    display_name: str
    builder_key: str
    backbone_name: str
    notes: str


#: Native input resolutions, keyed by timm model name (§4.1.4).
#:
#: These are not preferences. Swin-T tabulates relative-position biases for a
#: 7 x 7 window over a fixed token grid, so it accepts only its native 224 and the
#: backbone interpolates down to it. The other four are native at 384 and
#: therefore see the preprocessed radiograph with no interpolation at all, which
#: is why the data layer standardises on 384 x 384.
#:
#: Every name carries an explicit pretrained tag. An untagged timm name resolves
#: to a variant that may index no weights at all, in which case ``pretrained=True``
#: raises -- and §4.1.4 states plainly that all five X-ray encoders used
#: ImageNet-pretrained weights, so silently falling back to random init would
#: leave Table 4.5 describing a run that did not happen.
_XRAY_INPUT_SIZES: Dict[str, int] = {
    "convnextv2_tiny.fcmae_ft_in22k_in1k_384": 384,
    "maxvit_tiny_tf_384.in1k": 384,
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k": 224,
    "tf_efficientnetv2_s.in21k_ft_in1k": 384,
    "deit3_small_patch16_384.fb_in22k_ft_in1k": 384,
}


XRAY_CANDIDATES: Tuple[CandidateSpec, ...] = (
    CandidateSpec(
        candidate_id="xray_convnext_v2",
        modality="xray",
        display_name="ConvNeXt V2",
        builder_key="timm_xray",
        backbone_name="convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        notes=(
            "Table 4.5 row 1, the best-balanced X-ray candidate and the prior branch "
            "of the reported RCKF deployment route. FCMAE-pretrained then fine-tuned "
            "on ImageNet-22k/1k at 384^2, so it takes the preprocessed radiograph "
            "at native resolution."
        ),
    ),
    CandidateSpec(
        candidate_id="xray_maxvit",
        modality="xray",
        display_name="MaxViT",
        builder_key="timm_xray",
        backbone_name="maxvit_tiny_tf_384.in1k",
        notes=(
            "Table 4.5 row 2. MaxViT Tiny at its native 384^2; its block and grid "
            "attention are tabulated for that resolution. Slightly weaker on average "
            "than ConvNeXt V2 but architecturally distinct, which is why §4.3 "
            "keeps it as the second X-ray branch for testing whether the fusion and "
            "routing modules generalise across backbone families."
        ),
    ),
    CandidateSpec(
        candidate_id="xray_swin_t",
        modality="xray",
        display_name="Swin-T",
        builder_key="timm_xray",
        backbone_name="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
        notes=(
            "Table 4.5 row 3. Swin-T with 4 x 4 patches and a 7 x 7 attention window. "
            "Its relative-position bias table is fixed to the 224^2 token grid, so it "
            "is the one X-ray candidate whose input is interpolated down from 384."
        ),
    ),
    CandidateSpec(
        candidate_id="xray_efficientnet_v2",
        modality="xray",
        display_name="EfficientNet V2",
        builder_key="timm_xray",
        backbone_name="tf_efficientnetv2_s.in21k_ft_in1k",
        notes=(
            "Table 4.5 row 4. EfficientNetV2-S, ImageNet-21k pretrained and fine-tuned "
            "on 1k, native at 384^2 so it receives the radiograph with no interpolation."
        ),
    ),
    CandidateSpec(
        candidate_id="xray_deit",
        modality="xray",
        display_name="DeiT",
        builder_key="timm_xray",
        backbone_name="deit3_small_patch16_384.fb_in22k_ft_in1k",
        notes=(
            "Table 4.5 row 5. DeiT3-Small at 384^2: 16 x 16 patches giving a 24 x 24 "
            "grid of 576 tokens before the class token. Weakest of the five on every "
            "metric, retained because the candidate pool's value is complementarity "
            "(Table 4.11's oracle bound), not per-model strength."
        ),
    ),
)


MRI_CANDIDATES: Tuple[CandidateSpec, ...] = (
    CandidateSpec(
        candidate_id="mri_simsiam_ssl",
        modality="mri",
        display_name="SimSiam SSL + fine-tune",
        builder_key="simsiam_mri",
        backbone_name="mri-encoder-v2",
        notes=(
            "Table 4.6 row 1. The locally defined MriEncoder pretrained with a local "
            "projector, predictor and symmetric negative cosine objective, then "
            "fine-tuned through the CORN head. Nothing is downloaded, so the "
            "pretrained flag does not apply to it."
        ),
    ),
    CandidateSpec(
        candidate_id="mri_swin3d",
        modality="mri",
        display_name="Swin-3D Transformer",
        builder_key="swin3d_mri",
        backbone_name="swin3d_t",
        notes=(
            "Table 4.6 row 2. torchvision Swin3D-T, randomly initialised, patch "
            "embedding adapted to the two MRI channels."
        ),
    ),
    CandidateSpec(
        candidate_id="mri_r3d18_attn",
        modality="mri",
        display_name="R3D18-attn",
        builder_key="r3d18_attn_mri",
        backbone_name="r3d_18_attn",
        notes=(
            "Table 4.6 row 3. R3D-18 with the locally implemented channel gate on the "
            "pooled descriptor; the ablation against row 4 isolates that gate."
        ),
    ),
    CandidateSpec(
        candidate_id="mri_r3d18",
        modality="mri",
        display_name="R3D18",
        builder_key="r3d18_mri",
        backbone_name="r3d_18",
        notes=(
            "Table 4.6 row 4. torchvision R3D-18, randomly initialised, stem adapted "
            "to the two MRI channels. The ungated control for row 3."
        ),
    ),
)


CANDIDATE_SPECS: Dict[str, CandidateSpec] = {
    spec.candidate_id: spec for spec in XRAY_CANDIDATES + MRI_CANDIDATES
}

XRAY_BASE_CANDIDATE_IDS: Tuple[str, ...] = tuple(spec.candidate_id for spec in XRAY_CANDIDATES)
MRI_BASE_CANDIDATE_IDS: Tuple[str, ...] = tuple(spec.candidate_id for spec in MRI_CANDIDATES)
BASE_CANDIDATE_IDS: Tuple[str, ...] = XRAY_BASE_CANDIDATE_IDS + MRI_BASE_CANDIDATE_IDS


def candidate_spec(candidate_id: str) -> CandidateSpec:
    """Look up one specification, listing the alternatives when the id is unknown."""

    try:
        return CANDIDATE_SPECS[candidate_id]
    except KeyError:
        raise KoaError(
            f"Unknown base candidate {candidate_id!r}. The catalogue holds "
            f"{list(BASE_CANDIDATE_IDS)}."
        ) from None


def build_base_candidate(
    candidate_id: str,
    *,
    model_cfg: ModelConfig,
    pretrained: Optional[bool] = None,
) -> Tuple[str, nn.Module]:
    """Build one base candidate, returning ``(modality, module)``.

    The tuple return is not decoration. ``modality`` -- ``"xray"`` or ``"mri"`` --
    is what the caller needs to decide which tensor to feed the module, what batch
    size to use, and whether the MRI loader has to run at all, and deriving it
    from the id prefix at each call site is exactly the kind of duplicated string
    test this catalogue exists to remove. Callers unpack the tuple; do not collapse
    it to the module.

    ``pretrained`` defaults to the branch's configured value --
    ``model_cfg.xray_pretrained`` for X-ray candidates and
    ``model_cfg.mri_pretrained`` for MRI -- and an explicit value overrides it.
    Tests pass ``pretrained=False`` so that constructing the whole catalogue never
    touches the network. ``mri_simsiam_ssl`` ignores the flag entirely: its
    encoder is defined in this package and has no weights to fetch.
    """

    spec = candidate_spec(candidate_id)
    feature_dim = int(model_cfg.feature_dim)
    num_classes = int(model_cfg.num_classes)
    if pretrained is None:
        pretrained = (
            model_cfg.xray_pretrained if spec.modality == "xray" else model_cfg.mri_pretrained
        )

    if spec.builder_key == "timm_xray":
        return "xray", XrayCandidate(
            feature_dim=feature_dim,
            num_classes=num_classes,
            candidate_id=spec.candidate_id,
            backbone=TimmXrayBackbone(
                spec.backbone_name,
                feature_dim=feature_dim,
                pretrained=bool(pretrained),
                input_size=_XRAY_INPUT_SIZES[spec.backbone_name],
            ),
        )
    if spec.builder_key == "simsiam_mri":
        return "mri", SimSiamMriCandidate(
            feature_dim=feature_dim,
            num_classes=num_classes,
            candidate_id=spec.candidate_id,
        )
    if spec.builder_key == "swin3d_mri":
        return "mri", MriCandidate(
            feature_dim=feature_dim,
            num_classes=num_classes,
            candidate_id=spec.candidate_id,
            encoder=Swin3DMriBackbone(feature_dim, bool(pretrained)),
        )
    if spec.builder_key == "r3d18_attn_mri":
        return "mri", MriCandidate(
            feature_dim=feature_dim,
            num_classes=num_classes,
            candidate_id=spec.candidate_id,
            encoder=R3D18AttentionMriBackbone(feature_dim, bool(pretrained)),
        )
    if spec.builder_key == "r3d18_mri":
        return "mri", MriCandidate(
            feature_dim=feature_dim,
            num_classes=num_classes,
            candidate_id=spec.candidate_id,
            encoder=R3D18MriBackbone(feature_dim, bool(pretrained)),
        )
    raise KoaError(
        f"{spec.candidate_id!r} names builder {spec.builder_key!r}, which "
        "build_base_candidate has no branch for."
    )


def describe_catalog() -> Dict[str, Any]:
    """A JSON-ready summary of the registry, for ``--dry-run`` plan output.

    Constructs nothing, so it costs no weight download and no allocation, and can
    be printed by a command that is about to exit without touching the data.
    """

    def describe(spec: CandidateSpec) -> Dict[str, Any]:
        row = asdict(spec)
        if spec.builder_key == "timm_xray":
            row["input_size"] = _XRAY_INPUT_SIZES[spec.backbone_name]
        return row

    return {
        "xray_candidates": [describe(spec) for spec in XRAY_CANDIDATES],
        "mri_candidates": [describe(spec) for spec in MRI_CANDIDATES],
        "counts": {
            "xray": len(XRAY_CANDIDATES),
            "mri": len(MRI_CANDIDATES),
            "total": len(BASE_CANDIDATE_IDS),
        },
    }
