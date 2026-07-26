"""On-disk cache for decoded MRI volumes.

§4.1.3 specifies that volumes are cached as high-precision
float32 arrays at the target dimensions ``d32_h384_w384``, and this module is
that cache. Without it -- a configuration key with nothing behind it -- NIfTI
decode plus a 3-D trilinear resample runs once per sample *per epoch* on the CPU,
which is what forces MRI to train at batch size 1 with gradient accumulation
rather than at a real batch size.

What is cached is the volume **after** preprocessing and **before** augmentation.
Caching post-augmentation output would freeze one random draw per pair for the
whole run, quietly turning the augmenter off; caching pre-preprocessing input
would save nothing, since the resample is the expensive half.

The cache is keyed by ``pair_key`` under a directory named for the target shape,
so changing ``[input]`` dimensions selects a different directory instead of
silently reusing volumes of the wrong size.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from koa_multimodal.config.paths import resolve
from koa_multimodal.config.schema import InputConfig, PreprocessingConfig
from koa_multimodal.core.errors import ContractError

PathLike = Union[str, Path]

#: A pair_key becomes a filename, so it has to be one. Anything outside this set
#: could escape the cache directory or alias two pairs onto one file.
_SAFE_KEY = re.compile(r"^[A-Za-z0-9._-]+$")

_CHANNELS = 2


class MriCache:
    """A shape-scoped float32 volume cache. Disabled instances are inert no-ops."""

    def __init__(
        self,
        root: PathLike,
        shape: Tuple[int, int, int],
        enabled: bool = True,
    ) -> None:
        if len(shape) != 3:
            raise ContractError(f"MRI cache shape must be (D, H, W), got {shape}")
        self.root = Path(root)
        self.shape = tuple(int(value) for value in shape)
        self.enabled = bool(enabled)
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._rejected = 0

    @classmethod
    def from_config(
        cls,
        preprocessing_cfg: PreprocessingConfig,
        input_cfg: InputConfig,
    ) -> "MriCache":
        """Build the cache the configuration describes, enabled or not."""

        return cls(
            resolve(preprocessing_cfg.mri_cache_root),
            input_cfg.mri_shape,
            enabled=preprocessing_cfg.mri_cache_enabled,
        )

    # -- layout ------------------------------------------------------------

    @property
    def directory(self) -> Path:
        """``<root>/d32_h384_w384`` -- the shape is part of the path, not of a sidecar."""

        depth, height, width = self.shape
        return self.root / f"d{depth}_h{height}_w{width}"

    @property
    def expected_shape(self) -> Tuple[int, int, int, int]:
        """``(2, D, H, W)``: the stacked (T2, R2) pair, not a single channel."""

        return (_CHANNELS,) + self.shape

    def path_for(self, pair_key: str) -> Path:
        if not _SAFE_KEY.match(str(pair_key)):
            raise ContractError(
                f"pair_key {pair_key!r} is not filename-safe and cannot be cached; "
                "the cache filename is the primary key."
            )
        return self.directory / f"{pair_key}.npy"

    # -- access ------------------------------------------------------------

    def load(self, pair_key: str) -> Optional[np.ndarray]:
        """Return the cached volume, or ``None`` on any miss.

        A file whose shape or dtype does not match the current target is counted
        as ``rejected`` and treated as a miss. Returning it would hand the model a
        volume from a different configuration, which is worse than a slow epoch.
        """

        if not self.enabled:
            return None
        path = self.path_for(pair_key)
        if not path.is_file():
            self._misses += 1
            return None
        try:
            volume = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            # Truncated by an interrupted run; the next store overwrites it.
            self._rejected += 1
            self._misses += 1
            return None
        if volume.shape != self.expected_shape or volume.dtype != np.float32:
            self._rejected += 1
            self._misses += 1
            return None
        self._hits += 1
        return volume

    def store(self, pair_key: str, volume: np.ndarray) -> Optional[Path]:
        """Write a volume to the cache. Returns the path, or ``None`` when disabled.

        The write goes to a per-process temporary file and is then renamed, so a
        run interrupted mid-write cannot leave a half-written array that a later
        run would read as valid.
        """

        if not self.enabled:
            return None
        array = np.asarray(volume, dtype=np.float32)
        if array.shape != self.expected_shape:
            raise ContractError(
                f"Refusing to cache a volume of shape {array.shape} under "
                f"{self.directory.name}, which holds {self.expected_shape}"
            )
        path = self.path_for(pair_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The temporary name has to keep the .npy extension: np.save appends one
        # otherwise, and the rename would then look for a file that never existed.
        temporary = path.with_suffix(f".tmp{os.getpid()}.npy")
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
        self._stores += 1
        return path

    # -- reporting ---------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Cache counters, for the training log and the run record."""

        lookups = self._hits + self._misses
        return {
            "enabled": self.enabled,
            "directory": str(self.directory),
            "shape": list(self.expected_shape),
            "hits": self._hits,
            "misses": self._misses,
            "stores": self._stores,
            "rejected": self._rejected,
            "hit_rate": (self._hits / lookups) if lookups else 0.0,
        }
