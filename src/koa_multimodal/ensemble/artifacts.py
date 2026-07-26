"""Prediction artifacts and the out-of-fold provenance gate.

This is the methodological spine of the ensemble layer, and the thing most easily
weakened by accident.

Anything that *trains* a selector or a stacker must consume genuinely out-of-fold
predictions. Ordinary validation predictions look identical -- same schema, same
shape, plausible metrics -- but a meta-learner fitted on them is fitted on data the
base models already saw, and every downstream number inherits the leak invisibly.
So the guard is structural rather than advisory:

* only one artifact schema exists, and it **requires** per-row ``subject_ids`` and
  ``fold_ids`` plus per-fold provenance;
* ``train_subject_hash`` is sha256 over the *sorted* subject ids of a fold's
  training partition, so a reader can recompute it from the fold manifest and
  verify cryptographically that fold-k predictions came from a model trained on
  exactly the non-fold-k subjects;
* ``load_artifacts(..., require_oof=True)`` rejects anything whose split is not
  ``"oof"``.

Aligned artifacts are matched by **row order**, not by a pair-key join. A join
would quietly reorder or drop rows on a partial mismatch; an order check fails
loudly, which is the correct behaviour when the alignment is what makes the
comparison meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from koa_multimodal.core.errors import ProvenanceError
from koa_multimodal.core.serialization import read_json, write_json
from koa_multimodal.stats.metrics import metric_summary

PathLike = Union[str, Path]

PREDICTION_ARTIFACT_SCHEMA = "prediction-artifact-v2"
NUM_CLASSES = 5


@dataclass(frozen=True)
class FoldProvenance:
    """Per-fold evidence that a set of predictions is genuinely out-of-fold."""

    fold_id: int
    checkpoint_hash: str
    train_subject_hash: str
    train_subject_count: int

    def to_payload(self) -> Dict[str, Any]:
        return {
            "foldId": self.fold_id,
            "checkpointHash": self.checkpoint_hash,
            "trainSubjectHash": self.train_subject_hash,
            "trainSubjectCount": self.train_subject_count,
        }


@dataclass(frozen=True)
class PredictionArtifact:
    """One candidate's predictions over one split, with provenance."""

    candidate_id: str
    split: str
    labels: List[int]
    predictions: List[int]
    probabilities: List[List[float]]
    pair_keys: List[str]
    subject_ids: List[str]
    fold_ids: List[int]
    candidate_order: List[str]
    fold_provenance: Dict[int, FoldProvenance] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    source_path: Optional[Path] = None
    schema_version: str = PREDICTION_ARTIFACT_SCHEMA

    @property
    def is_oof(self) -> bool:
        return self.split == "oof"

    @property
    def size(self) -> int:
        return len(self.labels)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "candidateId": self.candidate_id,
            "split": self.split,
            "candidateOrder": list(self.candidate_order),
            "labels": list(self.labels),
            "predictions": list(self.predictions),
            "probabilities": [list(row) for row in self.probabilities],
            "pairKeys": list(self.pair_keys),
            "subjectIds": list(self.subject_ids),
            "foldIds": list(self.fold_ids),
            "foldProvenance": {
                str(fold): provenance.to_payload()
                for fold, provenance in sorted(self.fold_provenance.items())
            },
            "metrics": dict(self.metrics),
        }

    def write(self, path: PathLike) -> Path:
        return write_json(self.to_payload(), path)


def _validate_rows(artifact: PredictionArtifact) -> None:
    size = artifact.size
    lengths = {
        "predictions": len(artifact.predictions),
        "probabilities": len(artifact.probabilities),
        "pairKeys": len(artifact.pair_keys),
        "subjectIds": len(artifact.subject_ids),
        "foldIds": len(artifact.fold_ids),
    }
    mismatched = {name: count for name, count in lengths.items() if count != size}
    if mismatched:
        raise ProvenanceError(
            f"{artifact.candidate_id}: row-count mismatch against {size} labels: {mismatched}"
        )
    if size == 0:
        raise ProvenanceError(f"{artifact.candidate_id}: artifact contains no rows")
    if len(set(artifact.pair_keys)) != size:
        raise ProvenanceError(f"{artifact.candidate_id}: pairKeys are not unique")
    if any(label not in range(NUM_CLASSES) for label in artifact.labels):
        raise ProvenanceError(f"{artifact.candidate_id}: KL labels must lie in [0, 4]")
    if any(value not in range(NUM_CLASSES) for value in artifact.predictions):
        raise ProvenanceError(f"{artifact.candidate_id}: KL predictions must lie in [0, 4]")
    if any(not subject for subject in artifact.subject_ids):
        raise ProvenanceError(f"{artifact.candidate_id}: contains an empty subjectId")

    for index, row in enumerate(artifact.probabilities):
        if len(row) != NUM_CLASSES:
            raise ProvenanceError(
                f"{artifact.candidate_id}: row {index} has {len(row)} probabilities, expected {NUM_CLASSES}"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in row):
            raise ProvenanceError(f"{artifact.candidate_id}: invalid probability at row {index}")
        if not math.isclose(sum(row), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ProvenanceError(
                f"{artifact.candidate_id}: probabilities at row {index} sum to {sum(row):.6f}, not 1"
            )

    if not artifact.candidate_order:
        raise ProvenanceError(f"{artifact.candidate_id}: candidateOrder is required")
    if artifact.candidate_id not in artifact.candidate_order:
        raise ProvenanceError(
            f"{artifact.candidate_id}: candidateOrder does not contain this candidate"
        )
    if len(set(artifact.candidate_order)) != len(artifact.candidate_order):
        raise ProvenanceError(f"{artifact.candidate_id}: candidateOrder contains duplicates")


def _validate_oof(artifact: PredictionArtifact) -> None:
    """The checks that distinguish a genuine OOF artifact from a relabelled one."""

    if any(fold < 0 for fold in artifact.fold_ids):
        raise ProvenanceError(f"{artifact.candidate_id}: OOF foldIds must be non-negative")

    # A subject appearing in two folds means the grouping was by row, not by
    # patient, and the leak-free guarantee is void.
    seen: Dict[str, int] = {}
    for subject, fold in zip(artifact.subject_ids, artifact.fold_ids):
        previous = seen.setdefault(subject, fold)
        if previous != fold:
            raise ProvenanceError(
                f"{artifact.candidate_id}: subject {subject} appears in folds "
                f"{previous} and {fold}; OOF folds must be subject-grouped"
            )

    for fold in sorted(set(artifact.fold_ids)):
        provenance = artifact.fold_provenance.get(fold)
        if provenance is None:
            raise ProvenanceError(f"{artifact.candidate_id}: no provenance for fold {fold}")
        if not provenance.checkpoint_hash or not provenance.train_subject_hash:
            raise ProvenanceError(
                f"{artifact.candidate_id}: fold {fold} provenance is missing a hash"
            )


def validate_artifact(artifact: PredictionArtifact, *, require_oof: bool = False) -> None:
    _validate_rows(artifact)
    if require_oof and not artifact.is_oof:
        raise ProvenanceError(
            f"{artifact.candidate_id}: training a selector or stacker requires "
            f"split='oof', but this artifact is split={artifact.split!r}. Produce "
            "real out-of-fold predictions with per-fold retraining; do not relax "
            "this check."
        )
    if artifact.is_oof:
        _validate_oof(artifact)


def load_artifact(path: PathLike, *, require_oof: bool = False) -> PredictionArtifact:
    source = Path(path)
    payload = read_json(source)

    schema = str(payload.get("schemaVersion", ""))
    if schema != PREDICTION_ARTIFACT_SCHEMA:
        raise ProvenanceError(
            f"{source}: schemaVersion is {schema!r}, expected "
            f"{PREDICTION_ARTIFACT_SCHEMA!r}. Legacy artifacts carry no per-row "
            "subject or fold provenance and are not accepted."
        )

    provenance_payload = payload.get("foldProvenance") or {}
    artifact = PredictionArtifact(
        candidate_id=str(payload["candidateId"]),
        split=str(payload.get("split", "")),
        labels=[int(v) for v in payload["labels"]],
        predictions=[int(v) for v in payload["predictions"]],
        probabilities=[[float(x) for x in row] for row in payload["probabilities"]],
        pair_keys=[str(v) for v in payload.get("pairKeys", [])],
        subject_ids=[str(v) for v in payload.get("subjectIds", [])],
        fold_ids=[int(v) for v in payload.get("foldIds", [])],
        candidate_order=[str(v) for v in payload.get("candidateOrder", [])],
        fold_provenance={
            int(fold): FoldProvenance(
                fold_id=int(fold),
                checkpoint_hash=str(item.get("checkpointHash", "")),
                train_subject_hash=str(item.get("trainSubjectHash", "")),
                train_subject_count=int(item.get("trainSubjectCount", 0)),
            )
            for fold, item in provenance_payload.items()
        },
        metrics=payload.get("metrics") or {},
        source_path=source,
        schema_version=schema,
    )
    if not artifact.metrics:
        object.__setattr__(
            artifact, "metrics", metric_summary(artifact.labels, artifact.predictions)
        )
    validate_artifact(artifact, require_oof=require_oof)
    return artifact


def validate_aligned(
    artifacts: Sequence[PredictionArtifact], *, require_oof: bool = False
) -> None:
    """Every artifact must describe the same rows, in the same order."""

    if not artifacts:
        raise ProvenanceError("At least one prediction artifact is required")
    if len({artifact.candidate_id for artifact in artifacts}) != len(artifacts):
        raise ProvenanceError("Candidate ids must be unique within a pool")

    for artifact in artifacts:
        validate_artifact(artifact, require_oof=require_oof)

    reference = artifacts[0]
    for artifact in artifacts[1:]:
        for name, left, right in (
            ("labels", reference.labels, artifact.labels),
            ("pairKeys", reference.pair_keys, artifact.pair_keys),
            ("subjectIds", reference.subject_ids, artifact.subject_ids),
            ("foldIds", reference.fold_ids, artifact.fold_ids),
            ("candidateOrder", reference.candidate_order, artifact.candidate_order),
        ):
            if left != right:
                raise ProvenanceError(
                    f"{artifact.candidate_id}: {name} do not match {reference.candidate_id} "
                    "row for row. Aligned artifacts are compared by order, not joined."
                )

    if require_oof:
        loaded_order = [artifact.candidate_id for artifact in artifacts]
        if reference.candidate_order != loaded_order:
            raise ProvenanceError(
                f"Loaded order {loaded_order} does not match the recorded "
                f"candidateOrder {reference.candidate_order}"
            )


def load_artifacts(
    paths: Sequence[PathLike], *, require_oof: bool = False
) -> List[PredictionArtifact]:
    """Load a pool, reordering to the recorded ``candidateOrder`` when possible."""

    artifacts = [load_artifact(path, require_oof=require_oof) for path in paths]
    order = artifacts[0].candidate_order if artifacts else []
    by_id = {artifact.candidate_id: artifact for artifact in artifacts}
    if order and set(order) == set(by_id):
        artifacts = [by_id[candidate_id] for candidate_id in order]
    validate_aligned(artifacts, require_oof=require_oof)
    return artifacts


def probability_tensor(artifacts: Sequence[PredictionArtifact]) -> torch.Tensor:
    """``[candidates, samples, classes]``."""

    return torch.tensor(
        [artifact.probabilities for artifact in artifacts], dtype=torch.float32
    )
