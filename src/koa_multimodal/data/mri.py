"""MRI loading: two NIfTI volumes to one ``(2, D, H, W)`` float32 array.

``nibabel`` is an optional dependency. When it is absent this module falls back to
a self-contained NIfTI-1 reader, which is a complete substitute rather than a
degradation -- it honours the header's endianness, datatype and
``scl_slope``/``scl_inter`` scaling, so the array it returns is the array
``nibabel`` would have returned. That is why the fallback is silent: there is no
numerical difference for a caller to be warned about.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from koa_multimodal.config.schema import PreprocessingConfig
from koa_multimodal.core.errors import DataLayoutError
from koa_multimodal.data.preprocess import resize_linear_3d

PathLike = Union[str, Path]

#: NIfTI-1 ``datatype`` codes to NumPy type strings. Restricted to the codes the
#: OAI volumes actually use; an unlisted code raises rather than guessing a width.
_NIFTI_DTYPES = {
    2: "u1",
    4: "i2",
    8: "i4",
    16: "f4",
    64: "f8",
    512: "u2",
    768: "u4",
}

_NIFTI_HEADER_BYTES = 348
_NIFTI_MIN_BYTES = 352
_GZIP_MAGIC = b"\x1f\x8b"

#: Channel order of the stacked volume. Load-bearing: the fusion encoders, the
#: MRI augmenter's per-channel jitter and every saved checkpoint assume it, and
#: swapping the two would train silently against transposed physics.
CHANNEL_ORDER: Tuple[str, str] = ("t2", "r2")


# ------------------------------------------------------------------ reading


def read_nifti(path: PathLike) -> np.ndarray:
    """Read a NIfTI volume as float32, using ``nibabel`` when it is installed."""

    volume_path = Path(path)
    if not volume_path.is_file():
        raise DataLayoutError(f"MRI volume does not exist: {volume_path}")
    via_nibabel = _read_with_nibabel(volume_path)
    if via_nibabel is not None:
        return np.squeeze(via_nibabel).astype(np.float32)
    return _read_nifti1(volume_path)


def _read_with_nibabel(path: Path) -> Optional[np.ndarray]:
    try:
        import nibabel
    except ImportError:
        return None
    return np.asarray(nibabel.load(str(path)).get_fdata(dtype=np.float32))


def _read_nifti1(path: Path) -> np.ndarray:
    """Self-contained NIfTI-1 reader for single-file ``.nii`` / ``.nii.gz`` volumes."""

    payload = path.read_bytes()
    if payload[:2] == _GZIP_MAGIC:
        payload = gzip.decompress(payload)
    if len(payload) < _NIFTI_MIN_BYTES:
        raise DataLayoutError(f"NIfTI payload is too small to hold a header: {path}")

    # sizeof_hdr is always 348, so reading it wrong identifies the byte order.
    endian = "<"
    sizeof_hdr = struct.unpack("<i", payload[:4])[0]
    if sizeof_hdr != _NIFTI_HEADER_BYTES:
        sizeof_hdr = struct.unpack(">i", payload[:4])[0]
        endian = ">"
    if sizeof_hdr != _NIFTI_HEADER_BYTES:
        raise DataLayoutError(f"Not a NIfTI-1 file, or the header is corrupt: {path}")

    dims = struct.unpack(endian + "8h", payload[40:56])
    ndim = max(1, int(dims[0]))
    shape = tuple(int(value) for value in dims[1 : ndim + 1] if int(value) > 0)
    datatype = struct.unpack(endian + "h", payload[70:72])[0]
    offset = int(struct.unpack(endian + "f", payload[108:112])[0])
    slope = float(struct.unpack(endian + "f", payload[112:116])[0])
    intercept = float(struct.unpack(endian + "f", payload[116:120])[0])

    dtype_code = _NIFTI_DTYPES.get(datatype)
    if dtype_code is None:
        raise DataLayoutError(f"Unsupported NIfTI datatype code {datatype}: {path}")

    count = int(np.prod(shape))
    array = np.frombuffer(payload, dtype=np.dtype(endian + dtype_code), count=count, offset=offset)
    # NIfTI stores voxels fastest-axis-first, which is Fortran order.
    array = np.asarray(array, dtype=np.float32).reshape(shape, order="F")
    # A slope of 0 means "no scaling declared", not "scale everything to zero".
    if slope not in (0.0, 1.0):
        array = array * slope
    if intercept:
        array = array + intercept
    return np.squeeze(array).astype(np.float32)


# ------------------------------------------------------------------ loading


def load_mri_pair(
    t2_path: PathLike,
    r2_path: PathLike,
    *,
    shape: Tuple[int, int, int],
    preprocessing_cfg: PreprocessingConfig,
) -> np.ndarray:
    """Load the co-registered T2 and R2 volumes as one ``(2, D, H, W)`` float32 array.

    Channel order is ``(T2, R2)`` -- see :data:`CHANNEL_ORDER`.

    The two maps are quantitative and are bounded differently, so each gets its
    own clip window from the configuration: T2 relaxation times are clipped to
    ``[t2_clip_min, t2_clip_max]`` ms and divided by the ceiling, while R2 is
    already a rate in ``[0, 1]`` and is only clipped. Neither is ImageNet-
    normalised; the physical scale is the signal here.

    Clipping happens **before** the resample. Doing it afterwards would let an
    out-of-range voxel bleed into its trilinear neighbours first, so the clip
    would no longer bound what the encoder actually sees.
    """

    if len(shape) != 3:
        raise DataLayoutError(f"MRI target shape must be (D, H, W), got {shape}")
    depth, height, width = (int(value) for value in shape)

    t2 = _prepare_channel(
        read_nifti(t2_path),
        low=preprocessing_cfg.t2_clip_min,
        high=preprocessing_cfg.t2_clip_max,
        divide_by_high=True,
    )
    r2 = _prepare_channel(
        read_nifti(r2_path),
        low=preprocessing_cfg.r2_clip_min,
        high=preprocessing_cfg.r2_clip_max,
        divide_by_high=False,
    )
    t2 = resize_linear_3d(t2, depth, height, width)
    r2 = resize_linear_3d(r2, depth, height, width)
    return np.stack([t2, r2], axis=0).astype(np.float32)


def _prepare_channel(
    volume: np.ndarray,
    *,
    low: float,
    high: float,
    divide_by_high: bool,
) -> np.ndarray:
    """Promote to 3-D, replace non-finite voxels, then clip to the channel's window."""

    if volume.ndim != 3:
        volume = np.reshape(volume, (1,) + volume.shape[-2:])
    array = np.nan_to_num(
        volume.astype(np.float32, copy=False),
        nan=float(low),
        posinf=float(high),
        neginf=float(low),
    )
    array = np.clip(array, float(low), float(high))
    if divide_by_high:
        array = array / float(high)
    return array.astype(np.float32)
