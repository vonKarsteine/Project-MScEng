"""The paired dataset: index rows to batched tensors.

Two decisions here are worth stating explicitly, because both guard failure modes
that produce no error:

* **Augmentation has exactly one gate.** It runs when, and only when, the split is
  :attr:`Split.TRAIN`. Two gates -- an ``augment`` flag on the dataset and a
  ``split == "train"`` test inside the augmenter -- could disagree, and a split
  string the augmenter did not recognise would silently disable augmentation
  without contradicting anything.
* **Missing modality is a property of the loader, not of a sample.** See
  :func:`collate_pairs`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from koa_multimodal.config.schema import AugmentationConfig, InputConfig, PreprocessingConfig
from koa_multimodal.core.errors import ContractError
from koa_multimodal.core.seeding import sample_generator
from koa_multimodal.data.augment import augment_mri, augment_xray
from koa_multimodal.data.cache import MriCache
from koa_multimodal.data.mri import load_mri_pair
from koa_multimodal.data.records import PairRecord
from koa_multimodal.data.xray import load_xray_image, normalize_xray_imagenet_gray

PathLike = Union[str, Path]


class Split(str, Enum):
    """The four dataset roles.

    ``OOF`` is a real member rather than a bare string because the out-of-fold
    pipeline passes it around: as an unrecognised string it would fall through
    every ``== "train"`` comparison and disable augmentation by accident rather
    than by intent. Only ``TRAIN`` augments, so a dataset over held-out fold rows
    is correctly built as ``OOF``; the *training* half of a fold is built as
    ``TRAIN``.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    OOF = "oof"


def _as_split(value: Union[str, Split]) -> Split:
    if isinstance(value, Split):
        return value
    try:
        return Split(str(value))
    except ValueError:
        raise ContractError(
            f"Unknown split {value!r}; expected one of {[member.value for member in Split]}"
        ) from None


class KoaPairDataset(Dataset):
    """One (X-ray, MRI, KL grade) sample per index row.

    Randomness is per sample, not per worker: the generator for a row is derived
    from ``(seed, epoch, pair_key)``, so augmentation varies across epochs, is
    identical across runs, and does not depend on batch order, worker count or
    shuffling. Call :meth:`set_epoch` at the top of every epoch -- without it,
    every epoch replays the same draws.
    """

    def __init__(
        self,
        records: Sequence[PairRecord],
        *,
        data_root: PathLike,
        input_cfg: InputConfig,
        preprocessing_cfg: PreprocessingConfig,
        augmentation_cfg: AugmentationConfig,
        split: Union[str, Split],
        seed: int,
        epoch: int = 0,
        load_mri: bool = True,
        cache: Optional[MriCache] = None,
    ) -> None:
        self.records: List[PairRecord] = list(records)
        if not self.records:
            raise ContractError("KoaPairDataset was given no records")
        self.data_root = Path(data_root)
        self.input_cfg = input_cfg
        self.preprocessing_cfg = preprocessing_cfg
        self.augmentation_cfg = augmentation_cfg
        self.split = _as_split(split)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.load_mri = bool(load_mri)
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Advance the augmentation stream. Reproducible, not merely different."""

        self.epoch = int(epoch)

    @property
    def augmenting(self) -> bool:
        return self.split is Split.TRAIN

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        rng = sample_generator(self.seed, record.pair_key, self.epoch)

        xray, _provenance = load_xray_image(
            self.data_root / record.xray_relpath,
            size=self.input_cfg.xray_size,
            preprocessing_cfg=self.preprocessing_cfg,
        )
        mri = self._mri_for(record) if self.load_mri else None

        if self.augmenting:
            # X-ray draws first, so its stream is unaffected by whether the MRI
            # branch is loaded: an X-ray-only run and a fusion run augment the
            # same radiograph identically.
            xray = augment_xray(xray, self.augmentation_cfg.xray, rng)
            if mri is not None:
                mri = augment_mri(mri, self.augmentation_cfg.mri, rng)

        xray = normalize_xray_imagenet_gray(xray, self.preprocessing_cfg)
        return {
            "xray": torch.from_numpy(np.ascontiguousarray(xray)),
            "mri": None if mri is None else torch.from_numpy(np.ascontiguousarray(mri)),
            "label": torch.tensor(record.grade, dtype=torch.long),
            "pair_key": record.pair_key,
            "subject_id": record.subject_id,
        }

    def _mri_for(self, record: PairRecord) -> np.ndarray:
        """Decoded, preprocessed, pre-augmentation volume -- the cacheable form."""

        if self.cache is not None:
            cached = self.cache.load(record.pair_key)
            if cached is not None:
                return cached
        volume = load_mri_pair(
            self.data_root / record.t2_relpath,
            self.data_root / record.r2_relpath,
            shape=self.input_cfg.mri_shape,
            preprocessing_cfg=self.preprocessing_cfg,
        )
        if self.cache is not None:
            self.cache.store(record.pair_key, volume)
        return volume


def collate_pairs(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a batch, allowing an all-X-ray batch but rejecting a mixed one.

    Missing modality is a property of the **loader**, not of an individual sample:
    an X-ray-only stage builds its dataset with ``load_mri=False`` and every batch
    then carries ``mri=None``, which the fusion routes handle through the RCKF
    fallback. A batch mixing present and absent volumes has no defined meaning --
    the fallback is a per-batch branch, and silently zero-filling the missing rows
    would feed the measurement update a fabricated observation that looks
    confident. So it raises.
    """

    if not batch:
        raise ContractError("collate_pairs received an empty batch")
    volumes = [item["mri"] for item in batch]
    present = sum(volume is not None for volume in volumes)
    if present == 0:
        mri = None
    elif present == len(volumes):
        mri = torch.stack(volumes, dim=0)
    else:
        raise ContractError(
            f"A batch cannot mix present and missing MRI volumes "
            f"({present} of {len(volumes)} present). Missing modality is a property "
            "of the loader, not of a sample."
        )
    return {
        "xray": torch.stack([item["xray"] for item in batch], dim=0),
        "mri": mri,
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "pair_key": [item["pair_key"] for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
    }
