"""The pair index: one CSV row per (subject, knee), and the only source of truth.

Every split assignment, KL label and file location used anywhere in this package
comes from this one file. Three of its columns carry structural meaning:

* ``pair_key`` is the global primary key. OOF prediction artifacts, C-MODES route
  alignment and deployment records all identify a sample by it, so it must be
  unique across the whole index rather than merely within a split.
* ``subject_id`` is the grouping key for every leakage control -- split audits,
  out-of-fold assignment and the subject-clustered bootstrap. A patient
  contributes two knees, and those two rows are not independent samples.
* ``laterality`` is pure passthrough. It is carried for traceability into result
  records; nothing in the package branches on it.

The header is checked positionally rather than by name lookup: a column reordered
by a spreadsheet editor would otherwise be read silently and mislabel the dataset.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from koa_multimodal.core.errors import DataLayoutError

PathLike = Union[str, Path]

#: The 8 columns, in the order the index file must declare them.
INDEX_COLUMNS: Tuple[str, ...] = (
    "subject_id",
    "laterality",
    "split",
    "grade",
    "xray_relpath",
    "t2_relpath",
    "r2_relpath",
    "pair_key",
)

SPLITS: Tuple[str, ...] = ("train", "val", "test")

#: Kellgren-Lawrence grades KL0-KL4.
GRADES: Tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class PairRecord:
    """One (subject, knee) pair: a radiograph plus its two co-registered volumes."""

    subject_id: str
    laterality: str
    split: str
    grade: int
    xray_relpath: str
    t2_relpath: str
    r2_relpath: str
    pair_key: str


# ----------------------------------------------------------------------- read


def load_pair_index(
    path: PathLike,
    *,
    split: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[PairRecord]:
    """Read and validate the pair index.

    Validation runs over the whole file before ``split`` and ``limit`` are
    applied, so a narrowed view cannot pass while the index it came from is
    malformed -- in particular ``pair_key`` uniqueness is a property of the index,
    not of the subset a caller happens to request.
    """

    index_path = Path(path)
    if not index_path.is_file():
        raise DataLayoutError(f"Pair index does not exist: {index_path}")

    # utf-8-sig strips the byte-order mark a spreadsheet editor prepends, which
    # would otherwise corrupt the first header cell and fail the header check.
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise DataLayoutError(f"Pair index is empty: {index_path}") from None
        _validate_header(header, index_path)
        records = [
            _record_from_row(row, index_path, line)
            for line, row in enumerate(reader, start=2)
            if row
        ]

    _validate_unique_pair_keys(records, index_path)

    if split is not None:
        if split not in SPLITS:
            raise DataLayoutError(f"Unknown split {split!r}; expected one of {list(SPLITS)}")
        records = [record for record in records if record.split == split]
    if limit is not None:
        records = records[: max(0, int(limit))]
    return records


def _validate_header(header: Sequence[str], path: Path) -> None:
    found = tuple(name.strip() for name in header)
    if found != INDEX_COLUMNS:
        raise DataLayoutError(
            f"{path} must declare exactly the columns {list(INDEX_COLUMNS)} in that "
            f"order; found {list(found)}"
        )


def _record_from_row(row: Sequence[str], path: Path, line: int) -> PairRecord:
    if len(row) != len(INDEX_COLUMNS):
        raise DataLayoutError(
            f"{path} line {line}: expected {len(INDEX_COLUMNS)} fields, found {len(row)}"
        )
    values = dict(zip(INDEX_COLUMNS, (field.strip() for field in row)))

    split = values["split"]
    if split not in SPLITS:
        raise DataLayoutError(
            f"{path} line {line}: split {split!r} is not one of {list(SPLITS)}"
        )
    try:
        grade = int(values["grade"])
    except ValueError:
        raise DataLayoutError(
            f"{path} line {line}: grade {values['grade']!r} is not an integer"
        ) from None
    if grade not in GRADES:
        raise DataLayoutError(f"{path} line {line}: grade {grade} is outside KL{GRADES}")
    for column in ("subject_id", "pair_key", "xray_relpath"):
        if not values[column]:
            raise DataLayoutError(f"{path} line {line}: {column} is empty")

    return PairRecord(
        subject_id=values["subject_id"],
        laterality=values["laterality"],
        split=split,
        grade=grade,
        xray_relpath=values["xray_relpath"],
        t2_relpath=values["t2_relpath"],
        r2_relpath=values["r2_relpath"],
        pair_key=values["pair_key"],
    )


def _validate_unique_pair_keys(records: Sequence[PairRecord], path: Path) -> None:
    seen: Dict[str, int] = {}
    duplicates: List[str] = []
    for record in records:
        if record.pair_key in seen:
            duplicates.append(record.pair_key)
        seen[record.pair_key] = seen.get(record.pair_key, 0) + 1
    if duplicates:
        raise DataLayoutError(
            f"{path}: pair_key is the global primary key but {len(set(duplicates))} value(s) "
            f"repeat, e.g. {sorted(set(duplicates))[:5]}"
        )


# ---------------------------------------------------------------------- write


def write_pair_index(records: Iterable[PairRecord], path: PathLike) -> Path:
    """Write records to CSV in :data:`INDEX_COLUMNS` order. The one index writer."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(INDEX_COLUMNS)
        for record in records:
            writer.writerow([getattr(record, column) for column in INDEX_COLUMNS])
    return target


# -------------------------------------------------------------------- summary


def grade_distribution(records: Iterable[PairRecord]) -> Dict[int, int]:
    """Counts for every KL grade, including grades with zero samples.

    Dense over :data:`GRADES` so that a distribution table has the same five rows
    whatever subset it describes -- an absent KL4 in a fold is information, not a
    missing key.
    """

    counts = {grade: 0 for grade in GRADES}
    for record in records:
        counts[record.grade] += 1
    return counts


def subject_ids(records: Iterable[PairRecord]) -> List[str]:
    """The distinct subjects, sorted. Sorted because it feeds provenance hashes."""

    return sorted({record.subject_id for record in records})
