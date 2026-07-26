"""Train-split augmentation for both modalities.

Every random draw in this module comes from a :class:`numpy.random.Generator`
passed in by the caller. There is no module-level ``random``, no
``np.random.default_rng(None)``, and no internal seeding. Per-modality seeding is
the reproducibility hole this closes: were one path to take an optional seed
while the other drew its Gaussian noise from a fresh unseeded generator, one
modality would be reproducible and the other not, while the run record claimed a
single global seed for both.

Neither function checks the split. The gate lives in
:class:`~koa_multimodal.data.dataset.KoaPairDataset`, in one place, so val and
test are bit-exact no-ops by construction rather than by two agreeing conditions.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from koa_multimodal.config.schema import MriAugmentationConfig, XrayAugmentationConfig
from koa_multimodal.core.errors import DataLayoutError


# ------------------------------------------------------------------- X-ray


def augment_xray(
    image: np.ndarray,
    cfg: XrayAugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Geometric and photometric jitter for one ``(1, H, W)`` radiograph in ``[0, 1]``.

    Deliberately mild and anatomy-preserving: the crop keeps at least 82% of the
    area and rotation stays within +-12 degrees, because KL grading reads joint
    space width and osteophyte size, and an aggressive scale jitter would change
    the very quantity being graded. There is no vertical flip for the same reason
    -- a knee is not vertically symmetric.

    Output is clipped to ``[0, 1]``, which is why ImageNet-gray normalisation runs
    after this and not before.
    """

    if image.ndim != 3 or image.shape[0] != 1:
        raise DataLayoutError(f"Expected an X-ray tensor of shape (1, H, W), got {image.shape}")
    array = image[0].astype(np.float32, copy=True)
    height, width = array.shape

    array = _random_resized_crop(
        array, rng, cfg.random_resized_crop_scale, cfg.random_resized_crop_ratio
    )
    if rng.random() < cfg.horizontal_flip_p:
        array = np.ascontiguousarray(array[:, ::-1])
    if rng.random() < cfg.rotation_p:
        array = _rotate(array, float(rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees)))
    array = _affine_translate_scale(array, rng, cfg.affine_translate, cfg.affine_scale)
    array = _intensity_jitter(array, rng, cfg.intensity_brightness, cfg.intensity_contrast)
    if array.shape != (height, width):
        array = _resize_gray(array, height, width)
    return np.clip(array[None, :, :], 0.0, 1.0).astype(np.float32)


def _random_resized_crop(
    array: np.ndarray,
    rng: np.random.Generator,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
) -> np.ndarray:
    height, width = array.shape
    area = height * width
    for _ in range(10):
        target_area = area * float(rng.uniform(*scale))
        aspect = float(rng.uniform(*ratio))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        if 0 < crop_h <= height and 0 < crop_w <= width:
            top = int(rng.integers(0, height - crop_h, endpoint=True))
            left = int(rng.integers(0, width - crop_w, endpoint=True))
            cropped = array[top : top + crop_h, left : left + crop_w]
            return _resize_gray(cropped, height, width)
    return array


def _resize_gray(array: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
    resized = image.resize((width, height), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _rotate(array: np.ndarray, degrees: float) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.clip(array * 255.0, 0, 255).astype(np.uint8))
    rotated = image.rotate(degrees, resample=Image.Resampling.BICUBIC, fillcolor=0)
    return np.asarray(rotated, dtype=np.float32) / 255.0


def _affine_translate_scale(
    array: np.ndarray,
    rng: np.random.Generator,
    translate: float,
    scale: Tuple[float, float],
) -> np.ndarray:
    """Scale about the centre and shift, compositing onto a zero canvas."""

    height, width = array.shape
    factor = float(rng.uniform(*scale))
    scaled_h = max(1, int(round(height * factor)))
    scaled_w = max(1, int(round(width * factor)))
    scaled = _resize_gray(array, scaled_h, scaled_w)
    canvas = np.zeros((height, width), dtype=np.float32)
    max_dx = int(round(width * translate))
    max_dy = int(round(height * translate))
    center_y = (height - scaled_h) // 2 + int(rng.integers(-max_dy, max_dy, endpoint=True))
    center_x = (width - scaled_w) // 2 + int(rng.integers(-max_dx, max_dx, endpoint=True))
    dst_top = max(0, center_y)
    dst_left = max(0, center_x)
    src_top = max(0, -center_y)
    src_left = max(0, -center_x)
    copy_h = min(height - dst_top, scaled_h - src_top)
    copy_w = min(width - dst_left, scaled_w - src_left)
    if copy_h > 0 and copy_w > 0:
        canvas[dst_top : dst_top + copy_h, dst_left : dst_left + copy_w] = scaled[
            src_top : src_top + copy_h, src_left : src_left + copy_w
        ]
    return canvas


def _intensity_jitter(
    array: np.ndarray,
    rng: np.random.Generator,
    brightness: float,
    contrast: float,
) -> np.ndarray:
    """Contrast about the image mean, then a brightness offset. Left unclipped here."""

    offset = float(rng.uniform(-brightness, brightness))
    factor = float(rng.uniform(1.0 - contrast, 1.0 + contrast))
    mean = float(array.mean())
    return (array - mean) * factor + mean + offset


# --------------------------------------------------------------------- MRI


def augment_mri(
    volume: np.ndarray,
    cfg: MriAugmentationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip, per-channel intensity jitter, gamma and Gaussian noise for ``(2, D, H, W)``.

    The flips are in-plane only; the depth axis is never reordered, because slice
    order is anatomical and a reversed stack is not a plausible acquisition.
    Scale and shift are drawn **per channel**, since T2 and R2 are independent
    quantitative maps whose scanner-to-scanner drift is uncorrelated.
    """

    if volume.ndim != 4 or volume.shape[0] != 2:
        raise DataLayoutError(f"Expected an MRI tensor of shape (2, D, H, W), got {volume.shape}")
    array = volume.astype(np.float32, copy=True)

    if rng.random() < cfg.horizontal_flip_p:
        array = np.ascontiguousarray(array[:, :, :, ::-1])
    if rng.random() < cfg.vertical_flip_p:
        array = np.ascontiguousarray(array[:, :, ::-1, :])

    for channel in range(array.shape[0]):
        scale = float(rng.uniform(*cfg.channel_scale))
        shift = float(rng.uniform(*cfg.channel_shift))
        array[channel] = array[channel] * scale + shift

    gamma = float(rng.uniform(*cfg.gamma))
    array = np.power(np.clip(array, 0.0, 1.0), gamma)
    noise = rng.normal(0.0, cfg.gaussian_noise_std, size=array.shape).astype(np.float32)
    return np.clip(array + noise, 0.0, 1.0).astype(np.float32)
