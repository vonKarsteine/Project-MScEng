"""X-ray encoders. Both produce a ``[B, feature_dim]`` embedding from ``[B, 1, H, W]``.

**Resolution independence, mechanism one of two.** The data layer emits every
radiograph at a fixed ``1 x 384 x 384``, but the timm trunks were pretrained at
their own native resolutions, and two of the five -- Swin-T and MaxViT -- carry
window sizes and positional biases that make an arbitrary input size an outright
error rather than a quality question. :class:`TimmXrayBackbone` therefore resizes
the *input* to its own ``input_size``, and only when the incoming spatial size
differs. The condition is not defensive padding: EfficientNetV2-S and DeiT3-Small
run natively at 384 and so pass through bit-exact, paying no interpolation error
at all, while ConvNeXt V2 and Swin-T are downsampled to 224 and MaxViT to 256
(§4.1.4).

The MRI branch reaches the same property by the opposite arrangement:
:class:`~koa_multimodal.models.mri.R3D18MriBackbone` and
:class:`~koa_multimodal.models.mri.Swin3DMriBackbone` resample *unconditionally*
to ``(16, 112, 112)``, because nothing the data layer produces is ever already
that shape and a conditional test there could only ever be true. Both mechanisms
are deliberate. Rewriting either into the other's form is a behaviour change:
making the X-ray resize unconditional inserts a no-op interpolation into the two
native-384 candidates, and making the MRI resize conditional adds a branch that
never takes its other arm.

:class:`XrayBackbone` is a third case again -- it never resizes, and is
resolution-independent because it ends in a global ``AdaptiveAvgPool2d``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from koa_multimodal.core.errors import ConfigError


def timm_pretrained_available(model_name: str) -> bool:
    """Whether the installed timm release indexes real weights for this architecture.

    Not every architecture timm can *build* is one it can *initialise*: two of the
    five reported X-ray candidates, ``maxvit_pico_rw_256`` and ``efficientnetv2_s``,
    are defined without a weights entry in timm 1.0.24, so asking for pretraining
    is an error rather than a download. Checking the index first turns that into a
    message naming the candidate, instead of a library ``RuntimeError`` raised
    partway through building a training stage.
    """

    import timm

    cfg = timm.get_pretrained_cfg(model_name, allow_unregistered=True)
    return bool(getattr(cfg, "has_weights", False))


class TimmXrayBackbone(nn.Module):
    """A pretrained timm classification trunk reduced to a ``feature_dim`` embedding.

    The trunk is built with ``in_chans=1``, which is also what tells timm, when
    weights are being loaded, to collapse the pretrained RGB stem through its own
    ``adapt_input_conv`` -- a plain sum over the three filters, preserving the
    response a grayscale radiograph would have produced had it been replicated
    across RGB. That is the ``in_channels = 1`` case of the rule
    :mod:`koa_multimodal.models.inflate` generalises for the 3-D stems.
    ``num_classes=0`` with
    ``global_pool="avg"`` removes the ImageNet head and leaves a pooled feature
    vector, which a ``Linear`` + ``LayerNorm`` pair projects to the common 64-D
    representation every encoder in this package shares.

    The ``LayerNorm`` on the projection is what makes the embeddings of five
    architectures with wildly different feature scales -- ConvNeXt's 768-wide
    pooled map and DeiT3's 384-wide class token among them -- comparable enough
    for the RCKF measurement update to treat as the same quantity.

    Requesting ``pretrained=True`` raises when the weights cannot be had, whether
    because the installed timm carries none for the architecture or because the
    download fails. That is intentional, and asymmetric with the MRI backbones,
    which warn and fall back to random initialisation. ImageNet initialisation is
    part of the reported X-ray configuration -- §4.1.4 states all
    five encoders used it -- so silently losing it would leave Table 4.5 describing
    a run that did not happen. Random initialisation, by contrast, *is* the reported
    MRI configuration, so the MRI fallback lands on the documented default.
    """

    def __init__(
        self,
        model_name: str,
        feature_dim: int = 64,
        pretrained: bool = True,
        input_size: int = 384,
    ) -> None:
        super().__init__()
        import timm

        if input_size <= 0:
            raise ConfigError("TimmXrayBackbone.input_size must be positive")
        if feature_dim <= 0:
            raise ConfigError("TimmXrayBackbone.feature_dim must be positive")
        if pretrained and not timm_pretrained_available(model_name):
            raise ConfigError(
                f"timm {timm.__version__} indexes no pretrained weights for "
                f"{model_name!r}, so pretrained=True cannot be honoured. Falling back "
                "to random initialisation silently would contradict the dissertation's "
                "§4.1.4, which records every X-ray encoder as ImageNet-initialised. "
                "Either pin a timm release that ships these weights, select a variant "
                "that has them, or pass pretrained=False and report the candidate as "
                "randomly initialised."
            )

        self.backbone_name = str(model_name)
        self.input_size = int(input_size)
        self.feature_dim = int(feature_dim)
        self.pretraining_source = "ImageNet" if pretrained else "random"
        self.trunk = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="avg",
        )
        trunk_dim = int(getattr(self.trunk, "num_features", feature_dim))
        self.project = nn.Sequential(
            nn.Linear(trunk_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

    def forward(self, xray: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H, W] -> [B, feature_dim]``, resizing only when ``H, W`` differ.

        The trailing rank check covers the token-grid families, which return an
        unpooled map under some timm versions even with ``global_pool="avg"``
        configured; pooling it here costs nothing when the trunk already pooled.
        """

        if tuple(xray.shape[-2:]) != (self.input_size, self.input_size):
            xray = F.interpolate(
                xray,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        features = self.trunk(xray)
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.project(features)


class XrayBackbone(nn.Module):
    """A small hand-rolled convolutional encoder, used where timm cannot be.

    It exists so the synthetic smoke path, the contract tests, and any offline
    forward-pass check can construct a complete candidate without a weight
    download. It is not a reported candidate and appears in no results table.

    It never resizes: three stride-2 convolutions followed by
    ``AdaptiveAvgPool2d(1)`` make it accept any spatial size large enough to
    survive the stride, which is what lets a test run it on a 64 x 64 tensor and
    a smoke run feed it the full 384 x 384.

    ``BatchNorm2d`` is appropriate here and is *not* the choice
    :class:`~koa_multimodal.models.mri.MriEncoder` faces: the X-ray branch trains
    at batch size 4 over a 2-D activation map, so batch statistics are well
    populated, while the MRI branch trains at batch size 1 and uses ``GroupNorm``
    for that reason.
    """

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ConfigError("XrayBackbone.feature_dim must be positive")
        self.backbone_name = "xray_reference_cnn"
        self.feature_dim = int(feature_dim)
        self.pretraining_source = "random"
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )

    def forward(self, xray: torch.Tensor) -> torch.Tensor:
        """``[B, 1, H, W] -> [B, feature_dim]`` for any ``H, W`` above the stride."""

        return self.net(xray)
