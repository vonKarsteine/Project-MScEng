"""Radiograph loading: file on disk to a ``(1, size, size)`` float32 tensor in ``[0, 1]``.

The pipeline is fixed (§4.1.3):

    grayscale -> knee-centre square crop -> resize -> CLAHE

and ImageNet-gray normalisation is deliberately **not** part of it. See
:func:`normalize_xray_imagenet_gray` for why it has to run later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np

from koa_multimodal.config.schema import PreprocessingConfig
from koa_multimodal.core.errors import ConfigError, DataLayoutError
from koa_multimodal.data.preprocess import clahe_or_identity, knee_center_square_crop

PathLike = Union[str, Path]

#: The only locator implemented. Named in ``[preprocessing] xray_locator``.
IMPLEMENTED_LOCATOR = "edge_intensity_knee_center_square_crop"


def load_xray_image(
    path: PathLike,
    *,
    size: int,
    preprocessing_cfg: PreprocessingConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load one radiograph and return ``(image, provenance)``.

    ``image`` is ``(1, size, size)`` float32 in ``[0, 1]``; ``provenance`` records
    the crop geometry and whether CLAHE actually ran, which is what lets a
    deployment record state how a specific prediction's input was constructed --
    and lets a Grad-CAM overlay be mapped back onto the original radiograph.

    The uint8 round-trip before the resize is deliberate and load-bearing for
    reproducibility: PIL's bicubic resampler is what the recorded results were
    produced with, and it operates on 8-bit data.
    """

    if preprocessing_cfg.xray_locator != IMPLEMENTED_LOCATOR:
        raise ConfigError(
            f"preprocessing.xray_locator={preprocessing_cfg.xray_locator!r} is not "
            f"implemented; only {IMPLEMENTED_LOCATOR!r} exists."
        )
    image_path = Path(path)
    if not image_path.is_file():
        raise DataLayoutError(f"Radiograph does not exist: {image_path}")

    from PIL import Image

    with Image.open(image_path) as handle:
        array = np.asarray(handle.convert("L"), dtype=np.float32)
        array, crop = knee_center_square_crop(array)
        resized = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).resize(
            (int(size), int(size)), resample=Image.Resampling.BICUBIC
        )
        array = np.asarray(resized, dtype=np.float32)

    array, clahe_applied = clahe_or_identity(
        array,
        clip_limit=preprocessing_cfg.xray_clahe_clip_limit,
        tile_grid_size=preprocessing_cfg.xray_clahe_tile_grid_size,
        require=preprocessing_cfg.xray_require_clahe,
    )
    provenance = {
        "locator": IMPLEMENTED_LOCATOR,
        "clahe_applied": clahe_applied,
        "output_size": int(size),
        "crop": crop,
    }
    return array[None, :, :].astype(np.float32), provenance


def normalize_xray_imagenet_gray(
    image: np.ndarray,
    preprocessing_cfg: PreprocessingConfig,
) -> np.ndarray:
    """Standardise to the ImageNet grayscale statistics: ``(x - 0.449) / 0.226``.

    **Must run after augmentation, never before.** The X-ray augmenter finishes by
    clipping to ``[0, 1]``, which is correct for image data and destructive for
    standardised data: normalising first sends roughly half the pixels below zero,
    and the clip then flattens all of them to a single value. The ordering is a
    property of the pipeline, so this function is called once, at the end of
    :meth:`~koa_multimodal.data.dataset.KoaPairDataset.__getitem__`.

    The MRI branch is never ImageNet-normalised: the backbones are 3-D and either
    randomly initialised or dataset-pretrained, so ImageNet statistics carry no
    meaning for them.
    """

    if image.ndim != 3 or image.shape[0] != 1:
        raise DataLayoutError(f"Expected an X-ray tensor of shape (1, H, W), got {image.shape}")
    array = np.asarray(image, dtype=np.float32)
    mean = float(preprocessing_cfg.xray_imagenet_gray_mean)
    std = float(preprocessing_cfg.xray_imagenet_gray_std)
    return ((array - mean) / std).astype(np.float32)
