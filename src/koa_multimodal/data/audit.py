"""A read-only audit of the on-disk dataset against the index contract.

This is a *report*, not an assertion: it never raises for a finding, because the
answer "the index is missing and here is what else could not be checked" is more
useful than a traceback -- and because ``data/`` is legitimately empty in a
distributed working copy, where the audit still has to run and say so.

What it checks, in the order a problem would bite:

1. the index exists and parses under the 8-column contract;
2. per-split and per-grade counts, so a silently truncated index is visible;
3. **subject-level leakage across splits** -- one subject in two splits
   invalidates every generalisation claim in the dissertation, and no metric
   downstream can detect it;
4. that every referenced file is actually on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

from koa_multimodal.core.errors import DataLayoutError
from koa_multimodal.data.records import (
    SPLITS,
    PairRecord,
    grade_distribution,
    load_pair_index,
    subject_ids,
)

PathLike = Union[str, Path]

AUDIT_SCHEMA = "data-audit-v1"

_MODALITY_COLUMNS = ("xray_relpath", "t2_relpath", "r2_relpath")
_EXAMPLE_LIMIT = 10


@dataclass(frozen=True)
class AuditReport:
    """Everything the audit could establish, plus what it could not."""

    index_path: str
    data_root: str
    index_present: bool = False
    index_valid: bool = False
    data_root_present: bool = False
    sample_count: int = 0
    subject_count: int = 0
    split_counts: Dict[str, int] = field(default_factory=dict)
    grades: Dict[int, int] = field(default_factory=dict)
    grades_by_split: Dict[str, Dict[int, int]] = field(default_factory=dict)
    #: ``"train:test" -> [subject_id, ...]``. Non-empty means leakage.
    subject_leakage: Dict[str, List[str]] = field(default_factory=dict)
    #: ``"xray_relpath" -> count of rows whose file is absent``.
    missing_files: Dict[str, int] = field(default_factory=dict)
    missing_examples: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.index_present and self.index_valid and not self.errors

    def to_payload(self) -> Dict[str, Any]:
        """snake_case keys; the serialization boundary camelises them."""

        leakage = {
            pair: {"count": len(subjects), "subjects": subjects[:_EXAMPLE_LIMIT]}
            for pair, subjects in self.subject_leakage.items()
        }
        return {
            "schema_version": AUDIT_SCHEMA,
            "ok": self.ok,
            "index_path": self.index_path,
            "data_root": self.data_root,
            "index_present": self.index_present,
            "index_valid": self.index_valid,
            "data_root_present": self.data_root_present,
            "sample_count": self.sample_count,
            "subject_count": self.subject_count,
            "split_counts": dict(self.split_counts),
            "grades": dict(self.grades),
            "grades_by_split": {
                split: dict(counts) for split, counts in self.grades_by_split.items()
            },
            "subject_leakage": leakage,
            "missing_files": dict(self.missing_files),
            "missing_examples": list(self.missing_examples),
            "errors": list(self.errors),
        }


def audit_layout(data_root: PathLike, index_path: PathLike) -> AuditReport:
    """Audit ``index_path`` and the files it references under ``data_root``."""

    root = Path(data_root)
    index = Path(index_path)
    base = {
        "index_path": str(index),
        "data_root": str(root),
        "data_root_present": root.is_dir(),
    }

    if not index.is_file():
        return AuditReport(
            errors=[
                f"Pair index does not exist: {index}. Nothing else could be checked; "
                "data/ is empty in a fresh working copy."
            ],
            **base,
        )
    try:
        records = load_pair_index(index)
    except DataLayoutError as exc:
        return AuditReport(index_present=True, errors=[str(exc)], **base)

    errors: List[str] = []
    if not records:
        errors.append(f"{index} parsed but contains no rows")

    split_counts = {split: 0 for split in SPLITS}
    subjects_by_split: Dict[str, Set[str]] = {split: set() for split in SPLITS}
    for record in records:
        split_counts[record.split] += 1
        subjects_by_split[record.split].add(record.subject_id)

    leakage: Dict[str, List[str]] = {}
    for position, left in enumerate(SPLITS):
        for right in SPLITS[position + 1 :]:
            shared = sorted(subjects_by_split[left] & subjects_by_split[right])
            if shared:
                leakage[f"{left}:{right}"] = shared
                errors.append(
                    f"Subject leakage: {len(shared)} subject(s) appear in both the "
                    f"{left} and {right} splits, e.g. {shared[:5]}"
                )

    missing, examples = _missing_files(records, root)
    for column, count in missing.items():
        if count:
            errors.append(f"{count} row(s) reference a missing {column} under {root}")

    return AuditReport(
        index_present=True,
        index_valid=True,
        sample_count=len(records),
        subject_count=len(subject_ids(records)),
        split_counts=split_counts,
        grades=grade_distribution(records),
        grades_by_split={
            split: grade_distribution([r for r in records if r.split == split])
            for split in SPLITS
        },
        subject_leakage=leakage,
        missing_files=missing,
        missing_examples=examples,
        errors=errors,
        **base,
    )


def _missing_files(
    records: Sequence[PairRecord],
    root: Path,
) -> Tuple[Dict[str, int], List[str]]:
    """Count absent referenced files per modality column, keeping a few examples."""

    missing = {column: 0 for column in _MODALITY_COLUMNS}
    examples: List[str] = []
    for record in records:
        for column in _MODALITY_COLUMNS:
            relpath = getattr(record, column)
            # An empty relpath counts as missing: the index promises a paired row.
            if not relpath or not (root / relpath).is_file():
                missing[column] += 1
                if len(examples) < _EXAMPLE_LIMIT:
                    examples.append(f"{record.pair_key}:{column}={relpath or '<empty>'}")
    return missing, examples
