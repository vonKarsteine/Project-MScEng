"""The out-of-fold provenance gate -- §3.3.1.

The gate exists because ordinary validation predictions are indistinguishable from
out-of-fold ones by inspection: same schema, same shapes, plausible metrics. A
meta-learner fitted on them is fitted on data the base models already saw, and
every downstream number inherits the leak silently. So the checks are structural,
and ``require_oof`` is never relaxed.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from koa_multimodal.core.errors import ProvenanceError
from koa_multimodal.data.folds import train_subject_hash
from koa_multimodal.ensemble.artifacts import (
    FoldProvenance,
    PredictionArtifact,
    load_artifact,
    load_artifacts,
    validate_aligned,
    validate_artifact,
)

ROWS = 12
ORDER = ["route_a", "route_b"]


def make_artifact(candidate_id: str = "route_a", split: str = "oof", **overrides):
    torch.manual_seed(abs(hash(candidate_id)) % 1000)
    probabilities = torch.softmax(torch.randn(ROWS, 5), dim=1)
    fold_ids = [i % 3 for i in range(ROWS // 2) for _ in range(2)]
    payload = dict(
        candidate_id=candidate_id,
        split=split,
        labels=[i % 5 for i in range(ROWS)],
        predictions=[int(v) for v in probabilities.argmax(1)],
        probabilities=[[float(x) for x in row] for row in probabilities],
        pair_keys=["pair%02d" % i for i in range(ROWS)],
        subject_ids=["sub%02d" % (i // 2) for i in range(ROWS)],
        fold_ids=fold_ids,
        candidate_order=list(ORDER),
        fold_provenance={
            fold: FoldProvenance(fold, "ckpt%d" % fold, "subj%d" % fold, 4)
            for fold in set(fold_ids)
        },
    )
    payload.update(overrides)
    return PredictionArtifact(**payload)


class TestSchema:
    def test_a_well_formed_oof_artifact_validates(self):
        validate_artifact(make_artifact(), require_oof=True)

    def test_row_counts_must_agree(self):
        artifact = make_artifact(subject_ids=["sub00"])
        with pytest.raises(ProvenanceError, match="row-count mismatch"):
            validate_artifact(artifact)

    def test_probabilities_must_sum_to_one(self):
        bad = [[0.5] * 5 for _ in range(ROWS)]
        with pytest.raises(ProvenanceError, match="sum to"):
            validate_artifact(make_artifact(probabilities=bad))

    def test_pair_keys_must_be_unique(self):
        with pytest.raises(ProvenanceError, match="not unique"):
            validate_artifact(make_artifact(pair_keys=["dup"] * ROWS))

    def test_candidate_order_must_contain_this_candidate(self):
        with pytest.raises(ProvenanceError, match="candidateOrder"):
            validate_artifact(make_artifact(candidate_order=["someone_else"]))


class TestOofGate:
    def test_validation_predictions_are_rejected_for_training(self):
        """The whole point: a validation artifact must not train a meta-learner."""

        artifact = make_artifact(split="val")
        validate_artifact(artifact, require_oof=False)  # readable
        with pytest.raises(ProvenanceError, match="split='oof'"):
            validate_artifact(artifact, require_oof=True)

    def test_the_rejection_explains_the_remedy(self):
        with pytest.raises(ProvenanceError, match="per-fold retraining"):
            validate_artifact(make_artifact(split="val"), require_oof=True)

    def test_a_subject_may_not_cross_folds(self):
        """If it can, the grouping was by row and the guarantee is void."""

        crossing = [0, 1] + [2] * (ROWS - 2)
        with pytest.raises(ProvenanceError, match="subject-grouped"):
            validate_artifact(make_artifact(fold_ids=crossing))

    def test_every_fold_needs_provenance(self):
        with pytest.raises(ProvenanceError, match="no provenance for fold"):
            validate_artifact(make_artifact(fold_provenance={}))

    def test_empty_hashes_are_rejected(self):
        artifact = make_artifact()
        artifact.fold_provenance[0] = FoldProvenance(0, "", "", 0)
        with pytest.raises(ProvenanceError, match="missing a hash"):
            validate_artifact(artifact)


class TestTrainSubjectHash:
    def test_hash_is_reproducible_and_order_independent(self):
        """sha256 over the *sorted* subject ids, so a reader can recompute it from
        the fold manifest and verify the training partition cryptographically."""

        assert train_subject_hash(["b", "a", "c"]) == train_subject_hash(["c", "b", "a"])

    def test_hash_changes_when_the_partition_changes(self):
        assert train_subject_hash(["a", "b"]) != train_subject_hash(["a", "b", "c"])


class TestAlignment:
    def test_aligned_pool_validates(self):
        validate_aligned([make_artifact("route_a"), make_artifact("route_b")], require_oof=True)

    def test_row_order_mismatch_raises(self):
        """Artifacts are compared by order, never joined on pair key -- a join
        would silently reorder or drop rows on a partial mismatch."""

        b = make_artifact("route_b")
        shuffled = dataclasses.replace(b, pair_keys=list(reversed(b.pair_keys)))
        with pytest.raises(ProvenanceError, match="row for row"):
            validate_aligned([make_artifact("route_a"), shuffled])

    def test_label_mismatch_raises(self):
        b = make_artifact("route_b", labels=[0] * ROWS)
        with pytest.raises(ProvenanceError, match="labels"):
            validate_aligned([make_artifact("route_a"), b])

    def test_duplicate_candidate_ids_raise(self):
        with pytest.raises(ProvenanceError, match="unique"):
            validate_aligned([make_artifact("route_a"), make_artifact("route_a")])


class TestRoundTrip:
    def test_artifact_survives_write_and_read(self, tmp_path):
        artifact = make_artifact()
        path = artifact.write(tmp_path / "route_a.json")
        loaded = load_artifact(path, require_oof=True)
        assert loaded.candidate_id == artifact.candidate_id
        assert loaded.labels == artifact.labels
        assert loaded.fold_ids == artifact.fold_ids
        assert loaded.fold_provenance[0].train_subject_hash == "subj0"

    def test_legacy_schema_is_refused(self, tmp_path):
        """v7 emits and reads V2 only; a legacy artifact carries no per-row
        subject or fold provenance and cannot be audited."""

        from koa_multimodal.core.serialization import write_json

        path = write_json(
            {
                "candidateId": "old",
                "labels": [0, 1],
                "predictions": [0, 1],
                "probabilities": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            },
            tmp_path / "legacy.json",
        )
        with pytest.raises(ProvenanceError, match="schemaVersion"):
            load_artifact(path)

    def test_pool_loads_in_recorded_candidate_order(self, tmp_path):
        paths = [
            make_artifact(name).write(tmp_path / ("%s.json" % name))
            for name in reversed(ORDER)  # written out of order on purpose
        ]
        loaded = load_artifacts(paths, require_oof=True)
        assert [a.candidate_id for a in loaded] == ORDER
