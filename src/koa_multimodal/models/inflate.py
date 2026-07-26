"""Stem adaptation for the two-channel (T2, R2) volume.

This module is reachable only when ``model.mri_pretrained`` is true, and that flag
defaults to false on purpose. §4.1.4 states in as many words
that "the R3D18 and Swin3D-T MRI encoders were initialised without video-pretrained
weights and modified to accept the two MRI channels". Turning Kinetics-400 on makes
that sentence false, so this is an ablation switch and not a tuning knob: any run
that sets it must be reported as a departure from the documented configuration.

The problem it solves is narrow. A Kinetics-400 stem expects three RGB channels
and the paired knee volume supplies two. Dropping a channel or zero-padding to
three both change the magnitude of every pre-activation the pretrained trunk was
calibrated against, which de-calibrates the running statistics of every
normalisation layer downstream. The trunk still runs and still converges; it
simply produces features no better than -- often worse than -- random
initialisation, with nothing in the loss curve to indicate why.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from koa_multimodal.core.errors import ConfigError

#: ``mean_scaled`` is the only mode used by the reported configuration.
ADAPTATION_MODES: Tuple[str, ...] = ("mean_scaled", "slice", "random")


def adapt_conv_in_channels(
    conv: nn.Conv3d,
    in_channels: int,
    *,
    mode: str = "mean_scaled",
) -> nn.Conv3d:
    """Return a copy of ``conv`` accepting ``in_channels`` inputs.

    ``mean_scaled`` averages the pretrained input-channel filters and rescales by
    ``original_channels / in_channels``. The rescale is the part that matters.
    Without it, summing over two channels instead of three shrinks every
    pre-activation by a third. With it, for an input whose channels share a
    comparable mean, ``sum_c W'_c x_c`` reproduces the original three-channel
    response exactly, so the pretrained trunk sees the activation scale it was
    normalised for. This generalises the rule ``timm`` applies on the 2-D side:
    :func:`timm.models.adapt_input_conv` collapses an RGB stem to grayscale by
    *summing* the three filters, which is the ``in_channels = 1`` case of the same
    identity.

    ``slice`` keeps the first ``in_channels`` filters under the same rescale. It
    arbitrarily privileges whichever colour planes happen to come first and exists
    for ablation only. ``random`` reinitialises, discarding the pretrained stem;
    it is what the callers select when the trunk itself turned out to be randomly
    initialised, since copying random weights forward buys nothing.

    ``conv`` is left untouched -- the adapted layer is a fresh module on the same
    device and dtype, so a caller can compare the two.
    """

    if mode not in ADAPTATION_MODES:
        raise ConfigError(f"Stem adaptation mode must be one of {ADAPTATION_MODES}, got {mode!r}")
    if in_channels <= 0:
        raise ConfigError("Stem adaptation needs a positive in_channels")

    adapted = nn.Conv3d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)

    if mode == "random":
        return adapted

    with torch.no_grad():
        weight = conv.weight.detach()
        pretrained_channels = int(weight.shape[1])
        scale = float(pretrained_channels) / float(in_channels)
        if mode == "mean_scaled":
            source = weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1, 1)
        else:
            if pretrained_channels < in_channels:
                raise ConfigError(
                    f"slice adaptation needs at least {in_channels} pretrained input "
                    f"channels, the stem has {pretrained_channels}"
                )
            source = weight[:, :in_channels]
        adapted.weight.copy_(source * scale)
        if conv.bias is not None and adapted.bias is not None:
            adapted.bias.copy_(conv.bias.detach())
    return adapted
