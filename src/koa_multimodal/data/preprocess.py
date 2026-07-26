"""Array-level preprocessing primitives shared by the X-ray and MRI loaders.

Two different normalisations live here and are not interchangeable:

* :func:`normalize_float` is a **range** conversion -- 8-bit codes to ``[0, 1]``.
  It preserves relative intensity exactly and is what a caller wants when the
  image is already correctly exposed.
* :func:`percentile_normalize` is a **contrast** normalisation -- clip to the 1st
  and 99th percentiles, then rescale that window to ``[0, 1]``. It discards
  absolute intensity, which is precisely why the knee locator and the CLAHE stage
  use it: both compare intensities within one radiograph and must not depend on
  the detector's exposure. Nothing else should use it.

Everything here operates on NumPy arrays and returns float32.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import numpy as np

from koa_multimodal.core.errors import DataLayoutError, KoaWarning

_EPS = 1e-6


# ------------------------------------------------------------ normalisation


def normalize_float(image: np.ndarray) -> np.ndarray:
    """Convert an image to float32 in ``[0, 1]``, dividing by 255 when needed.

    An array whose maximum already lies at or below 1 is assumed to be scaled and
    is only clipped. Non-finite entries become 0 rather than propagating a NaN
    through the whole downstream batch.
    """

    array = np.asarray(image, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    if array.size and float(array.max()) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def percentile_normalize(image: np.ndarray, eps: float = _EPS) -> np.ndarray:
    """Clip to the 1st-99th percentile window and rescale that window to ``[0, 1]``.

    Robust to the handful of saturated pixels a radiograph carries at its
    collimator edges, which a min-max rescale would let dominate the whole range.
    A degenerate image -- all non-finite, or a window narrower than ``eps`` --
    returns zeros, so the locator downstream falls back to the geometric centre
    instead of dividing by nothing.
    """

    array = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(array[finite], [1.0, 99.0])
    array = np.clip(array, low, high)
    scale = float(high - low)
    if scale < eps:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - low) / scale).astype(np.float32)


# ------------------------------------------------------------------- CLAHE


def clahe_or_identity(
    image: np.ndarray,
    *,
    clip_limit: float,
    tile_grid_size: int,
    require: bool,
) -> Tuple[np.ndarray, bool]:
    """Contrast-limited adaptive histogram equalisation, or a clean degradation.

    Returns ``(image in [0, 1], clahe_applied)``.

    Normalisation happens **before** the optional CLAHE step and therefore happens
    unconditionally. The other order is a trap: a missing ``cv2`` would return the
    input untouched, still on a 0-255 scale, while every other path returned
    ``[0, 1]``. Nothing raises and nothing warns, so a model can be trained on
    inputs 255 times larger than the ones it is validated on. Making the range
    guarantee independent of whether an optional dependency imported is the point
    of this function.

    With ``require=True`` a missing ``cv2`` is fatal; otherwise it is a
    :class:`~koa_multimodal.core.errors.KoaWarning` and the normalised image is
    returned unequalised.
    """

    normalized = percentile_normalize(image)
    try:
        import cv2
    except ImportError:
        message = (
            "cv2 is not installed, so CLAHE was skipped. Intensities are still "
            "normalised to [0, 1], but local contrast differs from the recorded "
            "preprocessing and results are not comparable to the reported ones."
        )
        if require:
            raise DataLayoutError(
                message + " preprocessing.xray_require_clahe is true."
            ) from None
        warnings.warn(message, KoaWarning, stacklevel=2)
        return normalized, False

    quantized = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    grid = int(tile_grid_size)
    equalizer = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(grid, grid))
    return (equalizer.apply(quantized).astype(np.float32) / 255.0), True


# ------------------------------------------------------------- knee locator


def locate_knee_center(image: np.ndarray) -> Tuple[float, float]:
    """Estimate the knee centre as an edge-and-intensity weighted centroid.

    The joint is the brightest structured region of a knee radiograph, so the
    weight map adds two masked terms: bright pixels above the 55th intensity
    percentile, and strong gradients above the 70th edge percentile. Bone alone
    biases towards the femoral shaft; edges alone latch onto the collimator
    border. A blank or degenerate image falls back to the geometric centre.
    """

    if image.ndim != 2:
        raise DataLayoutError(f"Expected a 2-D X-ray image, got shape {image.shape}")
    gray = percentile_normalize(image)
    if not np.isfinite(gray).any() or float(gray.max()) <= _EPS:
        return (image.shape[0] / 2.0, image.shape[1] / 2.0)

    gy, gx = np.gradient(gray.astype(np.float32, copy=False))
    edge = np.sqrt(gx * gx + gy * gy)
    intensity_cut = np.percentile(gray, 55.0)
    edge_cut = np.percentile(edge, 70.0)
    weights = np.where(gray >= intensity_cut, gray, 0.0) + np.where(edge >= edge_cut, edge, 0.0)
    total = float(weights.sum())
    if total <= _EPS:
        return (image.shape[0] / 2.0, image.shape[1] / 2.0)

    yy, xx = np.indices(gray.shape, dtype=np.float32)
    return (
        float((yy * weights).sum() / total),
        float((xx * weights).sum() / total),
    )


def knee_center_square_crop(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crop the largest centred square around the located knee, edge-padding if it overhangs.

    The side is ``min(height, width)``, so the crop keeps as much of the joint as
    a square can hold and the subsequent resize never has to change aspect ratio.
    Edge padding rather than zero padding keeps the border intensity continuous,
    which matters because CLAHE runs after this and would otherwise equalise a
    black band into visible structure.

    The returned metadata is what makes the crop auditable from a result record:
    it is enough to map a heat-map coordinate back onto the original radiograph.
    """

    if image.ndim != 2:
        raise DataLayoutError(f"Expected a 2-D X-ray image, got shape {image.shape}")
    height, width = image.shape
    if height <= 0 or width <= 0:
        raise DataLayoutError(f"Invalid X-ray image shape: {image.shape}")

    center_y, center_x = locate_knee_center(image)
    side = max(1, min(height, width))
    top = int(round(center_y - side / 2.0))
    left = int(round(center_x - side / 2.0))
    bottom = top + side
    right = left + side

    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - height)
    pad_right = max(0, right - width)
    cropped = image[max(0, top) : min(height, bottom), max(0, left) : min(width, right)]
    if any((pad_top, pad_bottom, pad_left, pad_right)):
        cropped = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")
    if cropped.shape != (side, side):
        fixed = np.zeros((side, side), dtype=image.dtype)
        fixed[: cropped.shape[0], : cropped.shape[1]] = cropped
        cropped = fixed

    metadata = {
        "center_y": center_y,
        "center_x": center_x,
        "top": top,
        "left": left,
        "side": side,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "source_height": int(height),
        "source_width": int(width),
    }
    return cropped, metadata


# -------------------------------------------------------------- volume resize


def _resize_linear_axis(volume: np.ndarray, size: int, axis: int) -> np.ndarray:
    if size <= 0:
        raise DataLayoutError(f"Resize target must be positive, got {size}")
    old_size = volume.shape[axis]
    if old_size == size:
        return volume.astype(np.float32, copy=False)
    if old_size == 1:
        return np.repeat(volume, size, axis=axis).astype(np.float32)

    coords = np.linspace(0, old_size - 1, size, dtype=np.float32)
    lower = np.floor(coords).astype(np.int64)
    upper = np.clip(lower + 1, 0, old_size - 1)
    weight = coords - lower.astype(np.float32)
    lower_values = np.take(volume, lower, axis=axis)
    upper_values = np.take(volume, upper, axis=axis)
    weight_shape = [1] * volume.ndim
    weight_shape[axis] = size
    weights = weight.reshape(weight_shape)
    return (lower_values * (1.0 - weights) + upper_values * weights).astype(np.float32)


def resize_linear_3d(volume: np.ndarray, depth: int, height: int, width: int) -> np.ndarray:
    """Trilinear resample to ``(depth, height, width)``, one separable pass per axis.

    Separable because the three axes are resampled independently and the result is
    identical to the joint trilinear interpolation, at a fraction of the memory:
    the intermediate arrays never hold the full outer product of the three
    coordinate grids.
    """

    if volume.ndim != 3:
        raise DataLayoutError(f"Expected a 3-D volume, got shape {volume.shape}")
    resized = volume.astype(np.float32, copy=False)
    resized = _resize_linear_axis(resized, depth, axis=0)
    resized = _resize_linear_axis(resized, height, axis=1)
    resized = _resize_linear_axis(resized, width, axis=2)
    return resized.astype(np.float32)
