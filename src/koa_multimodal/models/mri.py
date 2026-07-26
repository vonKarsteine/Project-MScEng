"""MRI encoders. All produce a ``[B, feature_dim]`` embedding from ``[B, 2, D, H, W]``.

Channel order is ``(T2, R2)`` throughout, fixed by the data layer, and nothing
here re-orders it.

**Resolution independence, mechanism two of two.** The two torchvision video
trunks resample the volume *unconditionally* to ``(16, 112, 112)`` -- the shape
§4.1.4 records them as receiving -- because the data layer emits
``(32, 384, 384)`` and no input they will ever see is already at their native
size, so a conditional test could only take one arm. The X-ray branch does the
opposite for a matching reason: see the module docstring of
:mod:`koa_multimodal.models.xray`. :class:`MriEncoder`, being locally defined and
built for the native volume, never resizes at all and gets the same property from
``AdaptiveAvgPool3d``.

The reported configuration leaves both video trunks randomly initialised.
:mod:`koa_multimodal.models.inflate` documents what enabling Kinetics-400 costs.
"""

from __future__ import annotations

import warnings
from math import gcd
from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from koa_multimodal.core.errors import ConfigError, KoaWarning
from koa_multimodal.models.inflate import adapt_conv_in_channels

#: Architecture tag for :class:`MriEncoder`, bumped whenever a change invalidates
#: saved weights. Checkpoint grafting compares it and refuses a stale tag rather
#: than warm-starting from nothing: branch loads run with ``strict=False``, under
#: which a wholesale key mismatch is indistinguishable from success.
MRI_ENCODER_VERSION = "mri-encoder-v2"

#: What the torchvision video trunks are handed, after resampling.
VIDEO_TRUNK_INPUT_SIZE: Tuple[int, int, int] = (16, 112, 112)

#: The two MRI channels, in the order the data layer stacks them.
MRI_CHANNELS: Tuple[str, str] = ("t2", "r2")


def _group_norm(channels: int, groups: int) -> nn.GroupNorm:
    """``GroupNorm`` with a group count reduced to divide ``channels``.

    ``BatchNorm3d`` is unusable in this branch. An uncompressed two-channel
    ``32 x 384 x 384`` volume and its 3-D activation maps dominate memory, which
    pins the MRI batch size at 1, and a batch of one gives ``BatchNorm`` a
    zero-variance statistic per channel -- torch raises outright in training mode.
    """

    return nn.GroupNorm(gcd(groups, channels), channels)


class ResBlock3d(nn.Module):
    """A 3-D residual block with a projection shortcut on any shape change."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int],
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels, groups)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels, groups)
        self.activation = nn.GELU()
        if stride != (1, 1, 1) or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                _group_norm(out_channels, groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(volume)
        hidden = self.activation(self.norm1(self.conv1(volume)))
        hidden = self.norm2(self.conv2(hidden))
        return self.activation(hidden + residual)


class MriEncoder(nn.Module):
    """The locally defined anisotropic 3-D residual encoder, ``mri-encoder-v2``.

    Input ``[B, in_channels, D, H, W]``, nominally ``[B, 2, 32, 384, 384]``; output
    ``[B, feature_dim]``. This is the default MRI encoder, the trunk
    ``mri_simsiam_ssl`` pretrains and fine-tunes, and the branch the RCKF fusion
    routes take when no other encoder is named.

    The stride-``(1, 4, 4)`` ``7 x 7`` stem is the load-bearing design decision: it
    cuts in-plane resolution *before* any ``3 x 3 x 3`` kernel is applied at
    ``384^2``, which is what holds activations inside an 8 GB budget at batch
    size 1. Depth is preserved through the stem and the first stage -- the volume
    is only 32 slices and the informative anisotropy is in-plane, so spending
    depth early buys nothing -- then halved three times.

    ``net`` is deliberately a flat ``nn.Sequential`` whose element 0 is a bare
    ``Conv3d``. Quantisation-aware training indexes that position directly when
    it asserts the MRI stem was left out of the fake-quantised set, and wrapping
    the stem in a sub-block would break the assertion rather than the model.
    """

    def __init__(
        self,
        in_channels: int = 2,
        feature_dim: int = 64,
        stage_channels: Tuple[int, ...] = (32, 64, 96, 160, 224),
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if len(stage_channels) != 5:
            raise ConfigError("stage_channels must give the stem width plus four stage widths")
        if in_channels <= 0 or feature_dim <= 0:
            raise ConfigError("MriEncoder needs positive in_channels and feature_dim")
        stem, stage1, stage2, stage3, stage4 = (int(width) for width in stage_channels)

        self.backbone_name = MRI_ENCODER_VERSION
        self.encoder_version = MRI_ENCODER_VERSION
        self.pretraining_source = "random"
        self.in_channels = int(in_channels)
        self.feature_dim = int(feature_dim)
        self.stage_channels = tuple(int(width) for width in stage_channels)
        self.net = nn.Sequential(
            nn.Conv3d(
                self.in_channels,
                stem,
                kernel_size=(1, 7, 7),
                stride=(1, 4, 4),
                padding=(0, 3, 3),
                bias=False,
            ),
            _group_norm(stem, norm_groups),
            nn.GELU(),
            ResBlock3d(stem, stage1, stride=(1, 2, 2), groups=norm_groups),
            ResBlock3d(stage1, stage2, stride=(2, 2, 2), groups=norm_groups),
            ResBlock3d(stage2, stage3, stride=(2, 2, 2), groups=norm_groups),
            ResBlock3d(stage3, stage4, stride=(2, 2, 2), groups=norm_groups),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(stage4, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        """``[B, C, D, H, W] -> [B, feature_dim]`` at the volume's own resolution."""

        return self.net(mri)


class ChannelGate(nn.Module):
    """The locally implemented channel gate of the "R3D18-attn" row of Table 4.6.

    A bottlenecked two-layer MLP with a sigmoid, applied multiplicatively to the
    trunk's pooled feature vector -- squeeze-and-excitation moved from the spatial
    map to the pooled descriptor, which is where it can be added to a frozen-shape
    torchvision trunk without touching its internals. Gating after the global pool
    also keeps the ablation cheap: the gate sees 512 numbers per sample, not a 3-D
    map, so the parameter difference against plain R3D18 is small enough that any
    accuracy difference is attributable to the gating itself.

    The hidden width floors at 32 so the gate stays expressive at the narrow trunk
    widths this package works with.
    """

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(32, dim // max(1, reduction))
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.gate(features)


def _build_video_trunk(
    constructor: Callable[..., nn.Module],
    weights: Optional[Any],
    source: str,
) -> Tuple[nn.Module, str]:
    """Instantiate a torchvision video trunk, degrading audibly to random init.

    Asking for pretrained weights triggers a torch-hub download, which raises on
    an offline machine. Falling back is right for *this* branch and only this
    branch: random initialisation is its documented configuration, so the fallback
    lands on what the thesis describes rather than somewhere undescribed. It still
    has to be audible, because the run's model card would otherwise claim
    Kinetics-400 weights that were never loaded -- and :class:`KoaWarning` is
    promoted to an error under pytest, so a test that asks for pretraining and
    silently receives random weights fails instead of passing.
    """

    if weights is None:
        return constructor(weights=None), "random"
    try:
        return constructor(weights=weights), source
    except Exception as exc:  # noqa: BLE001 - reported, then recovered from
        warnings.warn(
            f"{getattr(constructor, '__name__', 'video trunk')}: could not obtain {source} "
            f"weights ({type(exc).__name__}); falling back to random initialisation. "
            "Record this run as randomly initialised.",
            KoaWarning,
            stacklevel=3,
        )
        return constructor(weights=None), "random"


class R3D18MriBackbone(nn.Module):
    """torchvision R3D-18 with a two-channel stem, projected to ``feature_dim``.

    The stem is *adapted* rather than replaced whenever real pretrained weights
    were obtained, so the Kinetics-400 filters survive the change of input
    channels; see :func:`~koa_multimodal.models.inflate.adapt_conv_in_channels`.
    When the trunk is randomly initialised -- the default, and the reported
    configuration -- the stem is reinitialised too, since copying random weights
    through an adaptation rule buys nothing.

    ``trunk.fc`` is replaced by ``Identity`` and the 512-D pooled descriptor goes
    through the same ``Linear`` + ``LayerNorm`` projection every other encoder
    uses, which is what puts all nine candidates in one comparable 64-D space.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        pretrained: bool = False,
        *,
        in_channels: int = 2,
        stem_adaptation: str = "mean_scaled",
    ) -> None:
        super().__init__()
        from torchvision.models.video import R3D_18_Weights, r3d_18

        if feature_dim <= 0:
            raise ConfigError("R3D18MriBackbone.feature_dim must be positive")

        self.backbone_name = "r3d_18"
        self.feature_dim = int(feature_dim)
        self.in_channels = int(in_channels)
        self.input_size = VIDEO_TRUNK_INPUT_SIZE
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        self.trunk, self.pretraining_source = _build_video_trunk(r3d_18, weights, "Kinetics-400")
        self.trunk.stem[0] = adapt_conv_in_channels(
            self.trunk.stem[0],
            self.in_channels,
            mode=stem_adaptation if self.pretraining_source != "random" else "random",
        )
        self.trunk_dim = int(self.trunk.fc.in_features)
        self.trunk.fc = nn.Identity()
        #: Populated only by :class:`R3D18AttentionMriBackbone`.
        self.gate: Optional[nn.Module] = None
        self.project = nn.Sequential(
            nn.Linear(self.trunk_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        """``[B, 2, D, H, W] -> [B, feature_dim]``, always via ``(16, 112, 112)``."""

        volume = F.interpolate(mri, size=self.input_size, mode="trilinear", align_corners=False)
        features = self.trunk(volume)
        if self.gate is not None:
            features = self.gate(features)
        return self.project(features)


class R3D18AttentionMriBackbone(R3D18MriBackbone):
    """R3D-18 plus the locally implemented channel gate: "R3D18-attn" in Table 4.6.

    Subclassing rather than flagging keeps the two rows of that table as two
    types, so a checkpoint cannot be loaded into the wrong one by accident: the
    gated variant's state dict carries ``gate.*`` keys the plain variant has no
    slot for, and the strict load refuses instead of quietly dropping them.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        pretrained: bool = False,
        *,
        in_channels: int = 2,
        gate_reduction: int = 4,
        stem_adaptation: str = "mean_scaled",
    ) -> None:
        super().__init__(
            feature_dim,
            pretrained,
            in_channels=in_channels,
            stem_adaptation=stem_adaptation,
        )
        self.backbone_name = "r3d_18_attn"
        self.gate = ChannelGate(self.trunk_dim, reduction=gate_reduction)


class Swin3DMriBackbone(nn.Module):
    """torchvision Swin3D-T with a two-channel patch embedding.

    The adapted layer is ``patch_embed.proj``, the ``Conv3d`` that tokenises the
    volume, and the same adaptation rule applies as for R3D-18. The window sizes
    and relative-position biases are left at the pinned library defaults, which is
    also why the ``(16, 112, 112)`` resample is unconditional: those biases are
    tabulated for a fixed token grid and do not follow the input around.
    """

    def __init__(
        self,
        feature_dim: int = 64,
        pretrained: bool = False,
        *,
        in_channels: int = 2,
        stem_adaptation: str = "mean_scaled",
    ) -> None:
        super().__init__()
        from torchvision.models.video import Swin3D_T_Weights, swin3d_t

        if feature_dim <= 0:
            raise ConfigError("Swin3DMriBackbone.feature_dim must be positive")

        self.backbone_name = "swin3d_t"
        self.feature_dim = int(feature_dim)
        self.in_channels = int(in_channels)
        self.input_size = VIDEO_TRUNK_INPUT_SIZE
        weights = Swin3D_T_Weights.KINETICS400_V1 if pretrained else None
        self.trunk, self.pretraining_source = _build_video_trunk(
            swin3d_t, weights, "Kinetics-400"
        )
        self.trunk.patch_embed.proj = adapt_conv_in_channels(
            self.trunk.patch_embed.proj,
            self.in_channels,
            mode=stem_adaptation if self.pretraining_source != "random" else "random",
        )
        self.trunk_dim = int(self.trunk.head.in_features)
        self.trunk.head = nn.Identity()
        self.project = nn.Sequential(
            nn.Linear(self.trunk_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

    def forward(self, mri: torch.Tensor) -> torch.Tensor:
        """``[B, 2, D, H, W] -> [B, feature_dim]``, always via ``(16, 112, 112)``."""

        volume = F.interpolate(mri, size=self.input_size, mode="trilinear", align_corners=False)
        return self.project(self.trunk(volume))
