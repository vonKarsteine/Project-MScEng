"""Out-of-fold assignment, and the hash that makes it verifiable.

This is the methodological spine of the ensemble chapter. The stacker and the
C-MODES selector are trained on the predictions of other models, so if any of
those predictions came from a model that had seen the sample it is predicting,
the selection statistics are contaminated and every downstream number is
optimistic by an unknown amount.

Two properties defend against that:

* **Subjects, not samples, are assigned to folds.** A patient contributes two
  knees whose radiographs correlate strongly, so splitting at sample level would
  leak one knee into the training set of the model that grades the other.
* **The assignment is provable after the fact.** :func:`train_subject_hash` is a
  sha256 over the *sorted* subject ids of a fold's training set. Recorded in the
  prediction artifact, it lets a reader recompute the hash from this manifest and
  confirm that fold-k predictions came from a model trained on exactly the
  non-fold-k subjects -- without trusting the pipeline that produced them.

This is the only fold implementation in the package, and deliberately so: a
second one, even a dead one, leaves a reader unable to tell which assignment the
recorded hashes were computed against.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Sequence, Set

from koa_multimodal.core.errors import ContractError
from koa_multimodal.data.records import GRADES, PairRecord

#: Recorded in every statistical payload as ``groupStratumRule``.
GROUP_STRATUM_RULE = "max_label_within_group"

FOLD_MANIFEST_SCHEMA = "oof-fold-manifest-v1"


# ------------------------------------------------------------------ hashing


def train_subject_hash(subject_ids: Iterable[str]) -> str:
    """sha256 over the newline-joined, **sorted**, de-duplicated subject ids.

    Sorted so the digest depends on the *set* of subjects and not on the order a
    particular run happened to enumerate them; that is what makes it reproducible
    from a manifest by a reader who never saw the run.
    """

    normalized = "\n".join(sorted({str(value) for value in subject_ids}))
    return sha256(normalized.encode("utf-8")).hexdigest()


# ------------------------------------------------------------- assignment


def grouped_stratified_fold_ids(
    records: Sequence[PairRecord],
    n_folds: int = 3,
    seed: int = 42,
    *,
    num_classes: int = len(GRADES),
) -> List[int]:
    """Assign each record a fold id, grouping by subject and stratifying by grade.

    Greedy rather than exhaustive, because exact grouped stratification is
    NP-hard: subjects are visited largest-first (most knees, then most
    concentrated grade histogram) and each is placed in the fold where it does the
    least harm. The first ``n_folds`` subjects seed one fold each, which stops the
    largest subjects piling into fold 0 before any counts exist to discriminate on.

    The score is the **marginal increase** in squared deviation from target that
    placing this subject would cause -- ``cost(after) - cost(before)`` -- not the
    projected fold's absolute deviation. The distinction is not cosmetic. Scoring
    the absolute level makes a fold that is already near target score near zero and
    keep winning, so folds fill close to sequentially instead of in rotation; on a
    3,536-row, 1,768-subject cohort at ``n_folds=3`` that yields roughly
    ``[2, 2348, 1186]``. Grouping stays sound either way -- no subject ever crosses
    a fold, so there is no leakage -- but a fold holding two samples contributes
    almost nothing to the out-of-fold prediction matrix, and the meta-learner
    trained on it would effectively see two folds, not three. Marginal cost makes
    an empty fold the cheapest destination, which is the behaviour the method
    describes.

    Size deviation enters the same score at weight 0.25, so a fold cannot balance
    its grade histogram by swallowing a disproportionate number of subjects.

    Ties are broken by a value drawn from ``random.Random(seed)``, so the result is
    deterministic given the seed and the record order, and does not depend on the
    insertion order of any hash table. :meth:`OOFFoldManifest.to_payload` publishes
    ``fold_sizes`` and ``fold_grade_counts`` so the realised balance is visible in
    the record rather than assumed.
    """

    if not records:
        raise ContractError("grouped_stratified_fold_ids received no records")

    grouped: Dict[str, List[int]] = {}
    for record in records:
        if not record.subject_id:
            raise ContractError("Every record needs a subject_id to be grouped by")
        if int(record.grade) not in range(num_classes):
            raise ContractError(f"Grade {record.grade} is outside [0, {num_classes - 1}]")
        histogram = grouped.setdefault(record.subject_id, [0] * num_classes)
        histogram[int(record.grade)] += 1

    folds = int(n_folds)
    if folds < 2 or folds > len(grouped):
        raise ContractError(
            f"n_folds must lie between 2 and the number of subjects ({len(grouped)}), got {folds}"
        )

    rng = random.Random(int(seed))
    tie_break = {subject_id: rng.random() for subject_id in grouped}
    ordered_subjects = sorted(
        grouped,
        key=lambda subject_id: (
            -sum(grouped[subject_id]),
            -max(grouped[subject_id]),
            tie_break[subject_id],
            subject_id,
        ),
    )
    total_by_class = [
        sum(histogram[class_id] for histogram in grouped.values()) for class_id in range(num_classes)
    ]
    target_by_class = [value / folds for value in total_by_class]
    target_size = len(records) / folds
    fold_class_counts = [[0] * num_classes for _ in range(folds)]
    fold_sizes = [0] * folds
    assignment: Dict[str, int] = {}

    for index, subject_id in enumerate(ordered_subjects):
        histogram = grouped[subject_id]
        if index < folds:
            chosen_fold = index
        else:
            subject_size = sum(histogram)
            scores = []
            for fold_id in range(folds):
                # Marginal cost: how much worse this fold gets, not how bad it is.
                class_score = sum(
                    (
                        (fold_class_counts[fold_id][class_id] + histogram[class_id]
                         - target_by_class[class_id]) ** 2
                        - (fold_class_counts[fold_id][class_id]
                           - target_by_class[class_id]) ** 2
                    )
                    / (target_by_class[class_id] + 1.0)
                    for class_id in range(num_classes)
                )
                size_score = (
                    (fold_sizes[fold_id] + subject_size - target_size) ** 2
                    - (fold_sizes[fold_id] - target_size) ** 2
                ) / (target_size + 1.0)
                scores.append((class_score + 0.25 * size_score, fold_sizes[fold_id], fold_id))
            chosen_fold = min(scores)[2]
        assignment[subject_id] = chosen_fold
        fold_sizes[chosen_fold] += sum(histogram)
        for class_id, count in enumerate(histogram):
            fold_class_counts[chosen_fold][class_id] += count

    return [assignment[record.subject_id] for record in records]


def subject_stratum(records: Sequence[PairRecord]) -> Dict[str, int]:
    """Map each subject to the stratum of its **highest** KL grade.

    A patient's two knees routinely carry different grades, so no label-pure
    clustering of subjects exists and some rule has to be chosen and recorded.
    The maximum is chosen because severity is what the stratification exists to
    preserve: a subject with a KL4 knee belongs in the severe stratum however mild
    the other knee is. The rule is recorded alongside every clustered-bootstrap
    result as ``groupStratumRule`` (:data:`GROUP_STRATUM_RULE`), so a reader can
    see that the strata are approximate and how.

    Used by the subject-clustered bootstrap, not by :func:`grouped_stratified_fold_ids`,
    which balances the full per-subject grade histogram and needs no collapse.
    """

    strata: Dict[str, int] = {}
    for record in records:
        current = strata.get(record.subject_id)
        grade = int(record.grade)
        if current is None or grade > current:
            strata[record.subject_id] = grade
    return strata


# -------------------------------------------------------------- manifest


@dataclass(frozen=True)
class OOFFoldManifest:
    """The fold assignment, in the form a prediction artifact can be checked against."""

    pair_keys: List[str]
    subject_ids: List[str]
    labels: List[int]
    fold_ids: List[int]
    folds: int
    seed: int

    @classmethod
    def from_records(
        cls,
        records: Sequence[PairRecord],
        *,
        n_folds: int = 3,
        seed: int = 42,
    ) -> "OOFFoldManifest":
        pair_keys = [record.pair_key for record in records]
        if len(set(pair_keys)) != len(pair_keys):
            raise ContractError("pair_key must be unique across the records being folded")
        return cls(
            pair_keys=pair_keys,
            subject_ids=[record.subject_id for record in records],
            labels=[int(record.grade) for record in records],
            fold_ids=grouped_stratified_fold_ids(records, n_folds=n_folds, seed=seed),
            folds=int(n_folds),
            seed=int(seed),
        )

    def held_out_subjects(self, fold_id: int) -> Set[str]:
        return {
            subject_id
            for subject_id, assigned in zip(self.subject_ids, self.fold_ids)
            if assigned == fold_id
        }

    def train_subjects(self, fold_id: int) -> Set[str]:
        """Every subject not held out by ``fold_id`` -- exactly who fold ``k`` trains on."""

        return set(self.subject_ids) - self.held_out_subjects(fold_id)

    def fold_sizes(self) -> List[int]:
        """Rows per fold. Published so an uneven assignment is visible in the record."""

        sizes = [0] * self.folds
        for fold_id in self.fold_ids:
            sizes[fold_id] += 1
        return sizes

    def fold_grade_counts(self) -> Dict[str, List[int]]:
        """Per-fold KL histogram, i.e. how well the stratification actually held."""

        counts = {str(fold_id): [0] * len(GRADES) for fold_id in range(self.folds)}
        for label, fold_id in zip(self.labels, self.fold_ids):
            counts[str(fold_id)][label] += 1
        return counts

    def to_payload(self) -> Dict[str, Any]:
        """Serialisable manifest, including the per-fold provenance hashes.

        Keys are snake_case; :func:`koa_multimodal.core.serialization.jsonable`
        camelises them at the write boundary.
        """

        provenance = {}
        for fold_id in range(self.folds):
            held_out = self.held_out_subjects(fold_id)
            train = set(self.subject_ids) - held_out
            provenance[str(fold_id)] = {
                "held_out_subject_hash": train_subject_hash(held_out),
                "train_subject_hash": train_subject_hash(train),
                "held_out_subject_count": len(held_out),
                "train_subject_count": len(train),
            }
        return {
            "schema_version": FOLD_MANIFEST_SCHEMA,
            "pair_keys": list(self.pair_keys),
            "subject_ids": list(self.subject_ids),
            "labels": list(self.labels),
            "fold_ids": list(self.fold_ids),
            "folds": self.folds,
            "seed": self.seed,
            "fold_sizes": self.fold_sizes(),
            "fold_grade_counts": self.fold_grade_counts(),
            "group_stratum_rule": GROUP_STRATUM_RULE,
            "fold_provenance": provenance,
        }
