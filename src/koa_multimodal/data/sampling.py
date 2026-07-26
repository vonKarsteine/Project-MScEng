"""Class-rebalanced sampling for the training split.

§4.1.3 specifies ``sqrt_balanced`` sampling. The OAI (Osteoarthritis
Initiative) paired cohort is dominated by KL0-KL2; KL3 and especially KL4 are
scarce, and uniform
sampling lets a run reach a respectable accuracy while never learning the severe
end of the scale -- which is the end that changes clinical management.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import WeightedRandomSampler

from koa_multimodal.core.errors import ContractError
from koa_multimodal.data.records import PairRecord


def sqrt_balanced_weights(records: Sequence[PairRecord]) -> List[float]:
    """Per-sample weights giving class ``k`` a sampling mass proportional to ``sqrt(n_k)``.

    Each sample of grade ``k`` is weighted ``1 / sqrt(n_k)``, so the class as a
    whole carries ``n_k / sqrt(n_k) = sqrt(n_k)``. That is the geometric mean of
    the two extremes: uniform sampling (mass ``n_k``, which never shows the model
    enough KL3/KL4) and fully balanced sampling (mass constant, which repeats the
    few KL4 knees so often that the model memorises those specific radiographs).
    The square root improves minority exposure while keeping the effective number
    of distinct images per class monotone in ``n_k``.

    Weights are scaled to average 1, so the expected number of draws per epoch is
    unchanged and the loss stays on the same scale as an unweighted run.
    """

    if not records:
        raise ContractError("sqrt_balanced_weights received no records")
    counts: Dict[int, int] = {}
    for record in records:
        counts[record.grade] = counts.get(record.grade, 0) + 1
    weights = [1.0 / sqrt(max(1, counts[record.grade])) for record in records]
    total = sum(weights)
    return [weight * len(weights) / total for weight in weights]


def make_sampler(
    records: Sequence[PairRecord],
    *,
    generator: Optional[torch.Generator] = None,
) -> WeightedRandomSampler:
    """A with-replacement sampler over ``sqrt_balanced_weights``, one epoch long.

    For the training split only. Applying it to val or test would resample the
    evaluation distribution and make every reported metric describe a cohort that
    does not exist.
    """

    weights = sqrt_balanced_weights(records)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
