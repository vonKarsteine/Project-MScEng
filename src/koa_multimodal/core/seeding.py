"""Deterministic seeding, including the per-sample generators augmentation needs.

The reproducibility gap this module closes: seeding the global RNGs is not enough
for a ``Dataset`` whose ``__getitem__`` draws random numbers, because worker
processes inherit an unspecified stream and any augmenter that calls
``np.random.default_rng(None)`` ignores the global seed entirely. Augmentation
here draws from a generator derived deterministically from
``(seed, epoch, pair_key)``, so a sample's augmentation is reproducible regardless
of worker count, batch order, or which modality is being augmented.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and torch, and pin cuDNN to deterministic kernels."""

    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def stable_hash(*parts: object) -> int:
    """A process-independent 64-bit hash.

    ``hash()`` is salted per process, so it cannot seed anything that has to be
    reproducible across runs. sha256 over the UTF-8 join is stable forever.
    """

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def sample_generator(seed: int, pair_key: str, epoch: int = 0) -> np.random.Generator:
    """A NumPy generator unique to one (sample, epoch) and reproducible from the seed.

    Threaded into both the X-ray and MRI augmenters so the two modalities share one
    reproducibility story instead of one being seeded and the other not.
    """

    return np.random.default_rng(stable_hash(seed, epoch, pair_key))


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None):
    """Give each DataLoader worker a distinct, reproducible RNG stream."""

    seed = (torch.initial_seed() if base_seed is None else base_seed) % (2**31)
    seed = (seed + worker_id) % (2**31)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
