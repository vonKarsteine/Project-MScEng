"""Typed configuration: one frozen dataclass per TOML section.

Defaults live *here*, on the dataclasses, and the TOML file only overrides them.
That gives every documented value exactly one authoritative, typed definition.

Two invariants are enforced by tests rather than convention
(``tests/test_config.py``):

* **Bijection.** Every key in ``configs/default.toml`` maps to a field, and every
  field appears in the file. No silent extras in either direction.
* **Every field has a consumer.** A field nothing reads is documentation
  pretending to be configuration. An unwired ``[cmodes]`` FedYogi block would
  leave the thesis constant beta_2 = 0.999 matching the code by coincidence
  rather than by configuration, and the file would still read as if it governed
  the run.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

from koa_multimodal.core.errors import ConfigError, KoaWarning


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "KOA_Multimodal_v7"
    methodology_version: str = "chapter3_v2"
    run_id: str = "run_001"

    def validate(self) -> None:
        if not self.run_id:
            raise ConfigError("project.run_id must be a non-empty result directory name")


@dataclass(frozen=True)
class DataConfig:
    training_root: str = "data/training"
    #: The single source of truth for every split, label and relative path.
    xray_index: str = "data/training/xray/index/pairs_00m.csv"

    def validate(self) -> None:
        if not self.xray_index.endswith(".csv"):
            raise ConfigError("data.xray_index must point at a CSV file")


@dataclass(frozen=True)
class InputConfig:
    xray_size: int = 384
    mri_depth: int = 32
    mri_height: int = 384
    mri_width: int = 384

    @property
    def mri_shape(self) -> Tuple[int, int, int]:
        return (self.mri_depth, self.mri_height, self.mri_width)

    def validate(self) -> None:
        for name in ("xray_size", "mri_depth", "mri_height", "mri_width"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"input.{name} must be positive")


@dataclass(frozen=True)
class PreprocessingConfig:
    xray_locator: str = "edge_intensity_knee_center_square_crop"
    xray_clahe_clip_limit: float = 2.0
    xray_clahe_tile_grid_size: int = 8
    #: Fail loudly at startup when cv2 is absent rather than silently skipping CLAHE.
    xray_require_clahe: bool = False
    xray_normalization: str = "imagenet_grayscale"
    xray_imagenet_gray_mean: float = 0.449
    xray_imagenet_gray_std: float = 0.226
    mri_dtype: str = "float32"
    #: §4.1.3 specifies a float32 volume cache at d32_h384_w384.
    mri_cache_enabled: bool = False
    mri_cache_root: str = "artifacts/cache/mri"
    t2_clip_min: float = 0.0
    t2_clip_max: float = 110.0
    r2_clip_min: float = 0.0
    r2_clip_max: float = 1.0

    def validate(self) -> None:
        if self.t2_clip_max <= self.t2_clip_min:
            raise ConfigError("preprocessing.t2_clip_max must exceed t2_clip_min")
        if self.r2_clip_max <= self.r2_clip_min:
            raise ConfigError("preprocessing.r2_clip_max must exceed r2_clip_min")
        if self.xray_imagenet_gray_std <= 0:
            raise ConfigError("preprocessing.xray_imagenet_gray_std must be positive")
        if self.xray_clahe_clip_limit <= 0:
            raise ConfigError("preprocessing.xray_clahe_clip_limit must be positive")
        if self.xray_clahe_tile_grid_size <= 0:
            raise ConfigError("preprocessing.xray_clahe_tile_grid_size must be positive")
        if self.mri_dtype != "float32":
            raise ConfigError("preprocessing.mri_dtype: only 'float32' is implemented")
        if self.xray_normalization != "imagenet_grayscale":
            raise ConfigError(
                "preprocessing.xray_normalization: only 'imagenet_grayscale' is implemented"
            )


@dataclass(frozen=True)
class XrayAugmentationConfig:
    random_resized_crop_scale: Tuple[float, float] = (0.82, 1.00)
    random_resized_crop_ratio: Tuple[float, float] = (0.90, 1.10)
    horizontal_flip_p: float = 0.5
    rotation_degrees: float = 12.0
    rotation_p: float = 0.85
    affine_translate: float = 0.04
    affine_scale: Tuple[float, float] = (0.95, 1.05)
    intensity_brightness: float = 0.08
    intensity_contrast: float = 0.12


@dataclass(frozen=True)
class MriAugmentationConfig:
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.25
    channel_scale: Tuple[float, float] = (0.92, 1.08)
    channel_shift: Tuple[float, float] = (-0.03, 0.03)
    gamma: Tuple[float, float] = (0.90, 1.10)
    gaussian_noise_std: float = 0.015


@dataclass(frozen=True)
class AugmentationConfig:
    xray: XrayAugmentationConfig = field(default_factory=XrayAugmentationConfig)
    mri: MriAugmentationConfig = field(default_factory=MriAugmentationConfig)

    def validate(self) -> None:
        probabilities = {
            "augmentation.xray.horizontal_flip_p": self.xray.horizontal_flip_p,
            "augmentation.xray.rotation_p": self.xray.rotation_p,
            "augmentation.mri.horizontal_flip_p": self.mri.horizontal_flip_p,
            "augmentation.mri.vertical_flip_p": self.mri.vertical_flip_p,
        }
        for name, value in probabilities.items():
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name} must lie in [0, 1]")
        ranges = {
            "augmentation.xray.random_resized_crop_scale": self.xray.random_resized_crop_scale,
            "augmentation.xray.random_resized_crop_ratio": self.xray.random_resized_crop_ratio,
            "augmentation.xray.affine_scale": self.xray.affine_scale,
            "augmentation.mri.channel_scale": self.mri.channel_scale,
            "augmentation.mri.channel_shift": self.mri.channel_shift,
            "augmentation.mri.gamma": self.mri.gamma,
        }
        for name, bounds in ranges.items():
            if len(bounds) != 2 or bounds[0] > bounds[1]:
                raise ConfigError(f"{name} must be an ascending [low, high] pair")
        if self.mri.gaussian_noise_std < 0:
            raise ConfigError("augmentation.mri.gaussian_noise_std must be non-negative")


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 5
    feature_dim: int = 64
    xray_pretrained: bool = True
    xray_pretrained_source: str = "ImageNet"
    #: §4.1.4 states 3-D backbones stay randomly initialised. Enabling
    #: Kinetics-400 pretrained weights makes that sentence false.
    mri_pretrained: bool = False
    mri_pretrained_source: str = "none_or_dataset_ssl"
    profiles: List[str] = field(
        default_factory=lambda: [
            "fusion_cmodes_oof",
            "fusion_traditional_ensemble",
            "rckf_cv2_simsiam",
            "xray_cmodes_oof",
        ]
    )

    def validate(self) -> None:
        if self.num_classes < 3:
            raise ConfigError("model.num_classes must be at least 3 for an ordinal head")
        if self.feature_dim <= 0:
            raise ConfigError("model.feature_dim must be positive")
        if not self.profiles:
            raise ConfigError("model.profiles must list at least one deployment profile")


@dataclass(frozen=True)
class RckfConfig:
    """§3.2. Symbols in comments match the thesis."""

    prior_variance: float = 0.15  # sigma_x^2
    variance_floor: float = 0.05  # R_floor
    variance_ceiling: float = 2.0  # R_ceil
    variance_groups: int = 1
    residual_weight: float = 0.1  # alpha_res, searched over {0.1, 0.2, 0.5}
    monotone_activation: str = "softplus"

    def validate(self) -> None:
        if self.prior_variance <= 0:
            raise ConfigError("rckf.prior_variance must be positive")
        if not 0 < self.variance_floor < self.variance_ceiling:
            raise ConfigError("rckf: require 0 < variance_floor < variance_ceiling")
        if self.variance_groups <= 0:
            raise ConfigError("rckf.variance_groups must be positive")
        if self.residual_weight < 0:
            raise ConfigError("rckf.residual_weight must be non-negative")
        if self.monotone_activation not in ("softplus", "gelu"):
            raise ConfigError("rckf.monotone_activation must be 'softplus' or 'gelu'")
        if self.monotone_activation == "gelu":
            warnings.warn(
                "rckf.monotone_activation='gelu' voids the monotonicity guarantee the "
                "variance estimator exists to provide (GELU dips below zero near "
                "x = -0.75). Ablation only.",
                KoaWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class CmodesConfig:
    """§3.3.

    ``selector_tau`` and ``selector_switch_cost`` are deliberately absent: they are
    *outputs* of out-of-fold calibration, not inputs, and live in the selector
    checkpoint's hyperparameters. They enter the routing rule only through their
    sum, so the identifiable free parameter is ``threshold = tau_s + c_switch``.
    """

    default_candidate_id: str = "rckf"
    risk_probability: str = "nll"
    risk_ordinal_weight: float = 1.0  # lambda in -log p(true) + lambda |y_hat - y| / 4
    boundary_bins: Tuple[float, ...] = (0.10, 0.30)
    selector_hidden_dim: int = 64
    selector_local_learning_rate: float = 1e-3
    selector_local_steps: int = 3
    selector_rounds: int = 5
    selector_server_learning_rate: float = 0.5
    selector_beta1: float = 0.9  # FedYogi beta_1
    selector_beta2: float = 0.999  # FedYogi beta_2
    selector_epsilon: float = 1e-3  # FedYogi tau

    def validate(self) -> None:
        if tuple(sorted(self.boundary_bins)) != tuple(self.boundary_bins):
            raise ConfigError("cmodes.boundary_bins must be sorted ascending")
        if not self.boundary_bins:
            raise ConfigError("cmodes.boundary_bins must not be empty")
        for name in ("selector_beta1", "selector_beta2"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ConfigError(f"cmodes.{name} must lie in [0, 1)")
        if self.selector_hidden_dim <= 0:
            raise ConfigError("cmodes.selector_hidden_dim must be positive")
        if self.selector_local_steps < 1 or self.selector_rounds < 1:
            raise ConfigError("cmodes.selector_local_steps and selector_rounds must be >= 1")
        if self.selector_epsilon <= 0:
            raise ConfigError("cmodes.selector_epsilon must be positive")
        if self.risk_probability != "nll":
            raise ConfigError("cmodes.risk_probability: only 'nll' is implemented")
        if self.risk_ordinal_weight < 0:
            raise ConfigError("cmodes.risk_ordinal_weight must be non-negative")


@dataclass(frozen=True)
class TrainingConfig:
    environment: str = "koa_project"
    device: str = "cuda:0"
    precision: str = "fp32"
    amp: bool = False
    seed: int = 42
    epochs: int = 20
    optimizer: str = "AdamW"
    learning_rate: float = 1e-4  # searched over [1e-5, 1e-4]
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    #: Windows: MRI decode happens in-process, and 0 keeps augmentation ordering
    #: deterministic. Per-sample seeding makes >0 safe, but 0 remains the default.
    num_workers: int = 0
    pin_memory: bool = True
    oof_folds: int = 3
    oof_seed: int = 42
    checkpoint_metric: str = "val_qwk"
    xray_batch_size: int = 4
    mri_batch_size: int = 1
    fusion_batch_size: int = 4
    multimodal_gradient_accumulation: int = 4
    contrastive_alignment_weight: float = 0.1  # lambda_align
    # alpha_res lives in [rckf] as rckf.residual_weight -- one home per constant.
    qat_kd_weight: float = 0.5  # lambda_1
    qat_boundary_weight: float = 0.2  # lambda_2
    qat_ordinal_weight: float = 0.1  # lambda_3
    qat_reliability_weight: float = 0.1  # lambda_4
    qat_selector_weight: float = 0.1  # lambda_5
    qat_temperature: float = 3.0  # tau
    qat_margin_epsilon: float = 0.05
    qat_selector_fallback_margin: float = 0.02

    def validate(self) -> None:
        if self.fusion_batch_size < 2:
            raise ConfigError(
                "training.fusion_batch_size must be >= 2: InfoNCE has no negative pairs "
                "at batch size 1, where bidirectional_info_nce returns a "
                "gradient-connected zero and the contrastive route silently degrades "
                "to plain CORN."
            )
        for name in ("xray_batch_size", "mri_batch_size", "multimodal_gradient_accumulation"):
            if getattr(self, name) < 1:
                raise ConfigError(f"training.{name} must be >= 1")
        if self.optimizer != "AdamW":
            raise ConfigError("training.optimizer: only 'AdamW' is implemented")
        if self.scheduler != "cosine":
            raise ConfigError("training.scheduler: only 'cosine' is implemented")
        if self.checkpoint_metric != "val_qwk":
            raise ConfigError("training.checkpoint_metric: only 'val_qwk' is implemented")
        if self.precision != "fp32":
            raise ConfigError("training.precision: results are recorded as fp32 only")
        if self.warmup_epochs >= self.epochs:
            raise ConfigError("training.warmup_epochs must be smaller than training.epochs")
        if self.oof_folds < 2:
            raise ConfigError("training.oof_folds must be at least 2")
        if self.qat_temperature <= 0:
            raise ConfigError("training.qat_temperature must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ConfigError("training.learning_rate must be positive, weight_decay >= 0")


@dataclass(frozen=True)
class StatisticsConfig:
    """§3.5."""

    bootstrap_iterations: int = 2000
    bootstrap_seed: int = 7
    bootstrap_alpha: float = 0.05
    cluster_by_subject: bool = True

    def validate(self) -> None:
        if self.bootstrap_iterations < 1000:
            raise ConfigError(
                "statistics.bootstrap_iterations must be at least 1000; the thesis "
                "reports B = 2000 and the resolvable p-value floor is 1/(B+1)."
            )
        if not 0.0 < self.bootstrap_alpha < 0.5:
            raise ConfigError("statistics.bootstrap_alpha must lie in (0, 0.5)")


@dataclass(frozen=True)
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    input: InputConfig = field(default_factory=InputConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rckf: RckfConfig = field(default_factory=RckfConfig)
    cmodes: CmodesConfig = field(default_factory=CmodesConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)

    def validate(self) -> "Config":
        for section in (
            self.project,
            self.data,
            self.input,
            self.preprocessing,
            self.augmentation,
            self.model,
            self.rckf,
            self.cmodes,
            self.training,
            self.statistics,
        ):
            section.validate()
        # Cross-section rules, which no single section can see.
        if self.rckf.variance_ceiling <= self.rckf.prior_variance:
            raise ConfigError(
                "rckf.variance_ceiling must exceed rckf.prior_variance, otherwise the "
                "missing-MRI fallback gain P/(P + R_ceil) would exceed the gain a live "
                "measurement can produce and a missing modality would look more "
                "trustworthy than a present one."
            )
        if self.model.feature_dim % self.rckf.variance_groups != 0:
            raise ConfigError(
                f"rckf.variance_groups ({self.rckf.variance_groups}) must divide "
                f"model.feature_dim ({self.model.feature_dim})"
            )
        return self
