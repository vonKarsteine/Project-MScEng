"""Shared fixtures.

Every test in this suite runs with **no dataset, no GPU and no network**. That is
the point: this checkout ships no data and no weights, so the suite has to be able
to establish correctness from contracts, synthetic tensors and dry runs alone.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List

import pytest
import torch

from koa_multimodal.config.loader import load_config
from koa_multimodal.config.paths import project_root
from koa_multimodal.data.records import PairRecord

#: Grade mix of the full paired cohort, Table 4.2 (train + val + test).
PAIRED_GRADE_MIX = (1385, 625, 963, 449, 114)
PAIRED_TOTAL = 3536


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def root() -> Path:
    return project_root()


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    random.seed(0)


def make_records(
    n_subjects: int = 60, grades_per_subject: int = 2, seed: int = 0
) -> List[PairRecord]:
    """A synthetic cohort with two knees per subject and a realistic grade mix."""

    rng = random.Random(seed)
    weights = list(PAIRED_GRADE_MIX)
    records: List[PairRecord] = []
    for subject_index in range(n_subjects):
        subject = "sub%04d" % subject_index
        for knee in range(grades_per_subject):
            grade = rng.choices(range(5), weights=weights)[0]
            side = "left" if knee == 0 else "right"
            key = "%s_%s" % (subject, side[0].upper())
            records.append(
                PairRecord(
                    subject_id=subject,
                    laterality=side,
                    split="train",
                    grade=grade,
                    xray_relpath="xray/%s.png" % key,
                    t2_relpath="mri/%s_t2.nii.gz" % key,
                    r2_relpath="mri/%s_r2.nii.gz" % key,
                    pair_key=key,
                )
            )
    return records


@pytest.fixture
def records() -> List[PairRecord]:
    return make_records()


@pytest.fixture
def small_batch():
    """X-ray and MRI tensors small enough to be fast, real enough to be meaningful."""

    return {
        "xray": torch.randn(4, 1, 96, 96),
        "mri": torch.randn(4, 2, 8, 48, 48),
        "labels": torch.tensor([0, 1, 2, 4]),
    }
