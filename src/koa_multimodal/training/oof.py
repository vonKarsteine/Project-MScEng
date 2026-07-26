"""Out-of-fold prediction generation -- §3.3.1.

This is the step whose integrity everything downstream rests on, and the one most
easily faked. A genuine out-of-fold prediction for fold *k* requires a model
**retrained from scratch on the other folds** -- relabelling a single validation
run as "oof" produces an artifact that is indistinguishable by inspection and
silently invalidates every selector, stacker and comparison built on it.

The plan is therefore emitted explicitly, with a per-fold subject hash computed
before any training happens, so a reviewer can check the partitioning of a run
that has not been executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from koa_multimodal.config.schema import Config
from koa_multimodal.data.folds import (
    grouped_stratified_fold_ids,
    train_subject_hash,
)
from koa_multimodal.data.records import PairRecord


@dataclass(frozen=True)
class FoldPlan:
    """One fold's partition, fully determined before any training runs."""

    fold_id: int
    train_subjects: List[str]
    holdout_subjects: List[str]
    train_rows: int
    holdout_rows: int
    holdout_grade_counts: Dict[int, int]
    train_subject_hash: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "foldId": self.fold_id,
            "trainSubjectCount": len(self.train_subjects),
            "holdoutSubjectCount": len(self.holdout_subjects),
            "trainRows": self.train_rows,
            "holdoutRows": self.holdout_rows,
            "holdoutGradeCounts": self.holdout_grade_counts,
            # sha256 over the sorted training subject ids. Recomputable by a
            # reader, which is what makes the leak-free claim verifiable rather
            # than asserted.
            "trainSubjectHash": self.train_subject_hash,
        }


@dataclass(frozen=True)
class OofPlan:
    n_folds: int
    seed: int
    total_rows: int
    total_subjects: int
    folds: List[FoldPlan]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schemaVersion": "oof-plan-v1",
            "nFolds": self.n_folds,
            "seed": self.seed,
            "totalRows": self.total_rows,
            "totalSubjects": self.total_subjects,
            "grouping": "subject",
            "stratification": "kl_grade",
            "folds": [fold.to_payload() for fold in self.folds],
        }

    def assert_leak_free(self) -> None:
        """No subject may appear in more than one holdout partition."""

        seen: Dict[str, int] = {}
        for fold in self.folds:
            for subject in fold.holdout_subjects:
                if subject in seen:
                    raise AssertionError(
                        f"subject {subject} is held out by folds {seen[subject]} and {fold.fold_id}"
                    )
                seen[subject] = fold.fold_id


def plan_oof(records: Sequence[PairRecord], config: Config) -> OofPlan:
    """Resolve the fold partition and its provenance hashes without training."""

    training = config.training
    fold_ids = grouped_stratified_fold_ids(
        list(records), n_folds=training.oof_folds, seed=training.oof_seed
    )

    subjects_by_fold: Dict[int, set] = {}
    rows_by_fold: Dict[int, int] = {}
    grades_by_fold: Dict[int, Dict[int, int]] = {}
    for record, fold in zip(records, fold_ids):
        subjects_by_fold.setdefault(fold, set()).add(record.subject_id)
        rows_by_fold[fold] = rows_by_fold.get(fold, 0) + 1
        counts = grades_by_fold.setdefault(fold, {})
        counts[int(record.grade)] = counts.get(int(record.grade), 0) + 1

    all_subjects = {record.subject_id for record in records}
    folds: List[FoldPlan] = []
    for fold in sorted(subjects_by_fold):
        holdout = sorted(subjects_by_fold[fold])
        train = sorted(all_subjects - subjects_by_fold[fold])
        folds.append(
            FoldPlan(
                fold_id=fold,
                train_subjects=train,
                holdout_subjects=holdout,
                train_rows=len(records) - rows_by_fold[fold],
                holdout_rows=rows_by_fold[fold],
                holdout_grade_counts=dict(sorted(grades_by_fold[fold].items())),
                train_subject_hash=train_subject_hash(train),
            )
        )

    plan = OofPlan(
        n_folds=training.oof_folds,
        seed=training.oof_seed,
        total_rows=len(records),
        total_subjects=len(all_subjects),
        folds=folds,
    )
    plan.assert_leak_free()
    return plan
