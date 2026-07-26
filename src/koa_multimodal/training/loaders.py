"""Dataset and DataLoader construction for a stage.

The batching story is a consequence of MRI cost. Table 4.3 reports a global batch
size of 16, but a 32x384x384x2 float32 volume decodes and resizes per sample, so
the physical MRI batch is 1 and the reported size is reached by gradient
accumulation instead. :meth:`StageSpec.effective_batch_size` reports the product,
so the two numbers never have to be reconciled by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from torch.utils.data import DataLoader

from koa_multimodal.config.paths import resolve
from koa_multimodal.config.schema import Config
from koa_multimodal.core.seeding import worker_init_fn
from koa_multimodal.data.cache import MriCache
from koa_multimodal.data.dataset import KoaPairDataset, Split, collate_pairs
from koa_multimodal.data.records import PairRecord, load_pair_index
from koa_multimodal.data.sampling import make_sampler
from koa_multimodal.training.stages import StageSpec


def load_split_records(
    config: Config, split: Split, *, sample_limit: Optional[int] = None
) -> List[PairRecord]:
    return load_pair_index(
        resolve(config.data.xray_index), split=split.value, limit=sample_limit
    )


def build_dataset(
    records: Sequence[PairRecord],
    config: Config,
    spec: StageSpec,
    split: Split,
    *,
    epoch: int = 0,
) -> KoaPairDataset:
    cache = MriCache(
        root=resolve(config.preprocessing.mri_cache_root),
        shape=config.input.mri_shape,
        enabled=config.preprocessing.mri_cache_enabled,
    )
    return KoaPairDataset(
        list(records),
        data_root=resolve(config.data.training_root),
        input_cfg=config.input,
        preprocessing_cfg=config.preprocessing,
        augmentation_cfg=config.augmentation,
        split=split,
        seed=config.training.seed,
        epoch=epoch,
        load_mri=spec.loads_mri,
        cache=cache,
    )


def build_loader(
    dataset: KoaPairDataset,
    config: Config,
    spec: StageSpec,
    split: Split,
) -> DataLoader:
    """Construct the loader for one split.

    Class-balanced sampling applies to training only. Validation and test must
    keep their natural class distribution, or the reported metrics describe a
    cohort that does not exist.
    """

    training = config.training
    is_train = split is Split.TRAIN
    sampler = make_sampler(dataset.records) if is_train else None
    return DataLoader(
        dataset,
        batch_size=spec.batch_size(config),
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        num_workers=training.num_workers,
        pin_memory=training.pin_memory,
        collate_fn=collate_pairs,
        drop_last=is_train and spec.modality.value == "fusion",
        worker_init_fn=(lambda worker_id: worker_init_fn(worker_id, training.seed))
        if training.num_workers > 0
        else None,
    )


def describe_loading(config: Config, spec: StageSpec) -> Dict[str, Any]:
    """The resolved data plan, for ``--dry-run`` output."""

    training = config.training
    return {
        "index": str(resolve(config.data.xray_index)),
        "trainingRoot": str(resolve(config.data.training_root)),
        "loadsMri": spec.loads_mri,
        "batchSize": spec.batch_size(config),
        "gradientAccumulation": spec.gradient_accumulation(config),
        "effectiveBatchSize": spec.effective_batch_size(config),
        "numWorkers": training.num_workers,
        "pinMemory": training.pin_memory,
        "amp": training.amp,
        "sampler": "sqrt_balanced (train split only)",
        "mriCacheEnabled": config.preprocessing.mri_cache_enabled,
        "mriCacheRoot": str(resolve(config.preprocessing.mri_cache_root)),
    }


def autocast_enabled(config: Config) -> bool:
    """Whether the training loop runs under automatic mixed precision.

    Off by default: §4.1.4 records that training used FP32 arithmetic, and
    ``precision`` is validated to ``fp32`` for exactly that reason. The
    mixed-precision path belongs to deployment, not to training.
    """

    return bool(config.training.amp)
