"""One record schema for every run, with ``status`` as the discriminating field.

A results record answers a single question: *what did this checkout actually
produce?* The honest answer in this working copy is "nothing yet", and the schema
has to be able to say that -- precisely, in a form a validator can check --
rather than leaving empty metric blocks that look indistinguishable from a run
that finished badly.

Hence :class:`RunStatus`. It is a field of the manifest, not an inference from
the shape of the tree, and every metric field in the record is required to agree
with it: populated throughout when ``complete``, ``None`` throughout when
``pending``. The second half of that rule is the load-bearing one. A pending
record carrying numbers is the exact signature of fabricated results, so
:mod:`koa_multimodal.records.validate` rejects it.

There is exactly **one** schema here and one validator. Two record shapes, with a
validator dispatching on whether a ``manifest.json`` happens to exist, is the
arrangement this replaces: the metric and oracle checks would live in only one of
those branches, so a record in the other shape would pass validation with empty
metric blocks and zero oracle records. The ``status`` field carries that variation
explicitly instead of letting the presence of a file carry it accidentally.

Numbers reported in the dissertation's Chapter 4 live in ``results/published/``, are
labelled as transcribed from the text, and must never be copied into a run
record. See ``results/published/SOURCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from koa_multimodal.core.serialization import camel, jsonable

#: Bumped when the on-disk record layout changes incompatibly.
RECORD_SCHEMA_VERSION = "koa-record-v1"


class RunStatus(str, Enum):
    """Whether the run behind this record has produced numbers.

    A ``str`` subclass so the value survives ``json.dumps`` and compares equal to
    the plain string a validator reads back out of the file.
    """

    PENDING = "pending"
    COMPLETE = "complete"


#: Metric field names, matching :mod:`koa_multimodal.stats.metrics` exactly, plus
#: the support count without which a recall is uninterpretable. One spelling of
#: each metric across the package: no ``val_acc``/``accuracy_val`` variants.
METRIC_FIELDS: Tuple[str, ...] = ("accuracy", "qwk", "kl1_recall", "sample_count")

#: The camelCase spellings as they appear on disk. The validator scans for these.
METRIC_KEYS: FrozenSet[str] = frozenset(camel(name) for name in METRIC_FIELDS)


# ------------------------------------------------------------- expected counts
#
# From Chapter 4. These are structural facts about the experimental
# design -- how many models were trained and over how many pairs -- not results,
# so they belong in a pending record: they describe what a run *would* do.

#: Table 4.5: ConvNeXt V2, MaxViT, Swin-T, EfficientNet V2, DeiT.
EXPECTED_XRAY_CANDIDATES = 5
#: Table 4.6: SimSiam SSL, Swin-3D, R3D18-attn, R3D18.
EXPECTED_MRI_CANDIDATES = 4
#: Table 4.7: late concatenation, gated, cross-attention.
EXPECTED_TRADITIONAL_FUSIONS = 3
#: Table 4.8: two X-ray backbones x two MRI backbones.
EXPECTED_CONTRASTIVE_FUSIONS = 4
#: Table 4.9: the same four pairings under RCKF.
EXPECTED_RCKF_FUSIONS = 4
#: Table 4.2: 2,506 training + 357 validation + 673 testing paired samples.
EXPECTED_PAIRED_TOTAL = 3536
#: Table 4.11: the X-ray-only and bimodal four-model selection ceilings.
EXPECTED_ORACLE_BOUNDS = 2

#: Total fusion models a full run trains, and the count Table 4.4 costs at 20
#: epochs each. Derived, so adding a route updates it in one place.
EXPECTED_FUSION_TOTAL = (
    EXPECTED_TRADITIONAL_FUSIONS + EXPECTED_CONTRASTIVE_FUSIONS + EXPECTED_RCKF_FUSIONS
)

#: Pool ids for the two oracle bounds of Table 4.11.
ORACLE_POOL_IDS: Tuple[str, ...] = ("xray_only_four_model", "bimodal_four_model")


#: Every file a record directory holds. The set is fixed rather than discovered,
#: so a run that silently failed to emit its statistical validation is a missing
#: file the validator names, not an absence nobody notices.
RECORD_FILES: Tuple[str, ...] = (
    "manifest.json",
    "candidates.json",
    "fusion.json",
    "ensemble.json",
    "oracle_bounds.json",
    "statistical_validation.json",
    "qat_deployment.json",
    "training_parameters.json",
)


@dataclass(frozen=True)
class MetricBlock:
    """The three headline metrics plus their support, on one split.

    Every field is ``Optional`` and every field is ``None`` in a pending record.
    That is deliberate: there is no zero-valued or sentinel default, because a
    ``0.0`` accuracy is a *result* -- a claim that the model got nothing right --
    and would be indistinguishable from "not measured" once written to disk.

    ``sample_count`` is not decoration. KL1 recall over the 119 KL1 samples of the
    test split and KL1 recall over a 12-sample slice are the same number with
    wildly different meaning, and a record that omits the denominator cannot be
    audited.
    """

    accuracy: Optional[float] = None
    qwk: Optional[float] = None
    kl1_recall: Optional[float] = None
    sample_count: Optional[int] = None

    @classmethod
    def pending(cls) -> "MetricBlock":
        """An explicitly unmeasured block. Named, so call sites read as intent."""

        return cls()

    @property
    def is_pending(self) -> bool:
        """True when nothing has been measured: every field is ``None``."""

        return all(getattr(self, name) is None for name in METRIC_FIELDS)

    @property
    def is_populated(self) -> bool:
        """True when every field carries a value.

        Deliberately not ``not is_pending``. A block with two of four fields set
        is neither pending nor populated -- it is a partially written record, and
        both properties returning ``False`` is what lets the validator say so.
        """

        return all(getattr(self, name) is not None for name in METRIC_FIELDS)

    def to_payload(self) -> Dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class CandidateRecord:
    """One model's line in Table 4.5, 4.6, 4.7, 4.8 or 4.9.

    Validation and test metrics are separate blocks because the two are read for
    different purposes -- validation QWK selects the checkpoint, test metrics are
    reported -- and collapsing them would let a selection number be quoted as a
    result.

    ``checkpoint_hash`` is what ties the row to a file under ``artifacts/``. It is
    ``None`` while pending, and it is the only field here that can prove the
    numbers beside it came from a specific set of weights.
    """

    candidate_id: str
    modality: str
    display_name: str
    validation: MetricBlock = field(default_factory=MetricBlock)
    test: MetricBlock = field(default_factory=MetricBlock)
    checkpoint_hash: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class OracleRecord:
    """A selection ceiling: Table 4.11.

    The oracle picks, per sample, the best of ``candidate_ids`` using the true
    label. It is not achievable -- it is the upper bound any router over that pool
    could reach, and the gap between it and the C-MODES row of Table 4.10 is the
    headroom the dissertation argues about. Storing ``candidate_ids`` rather than
    a pool name keeps the bound meaningful: the same name over a different pool is
    a different ceiling.
    """

    pool_id: str
    candidate_ids: List[str]
    metrics: MetricBlock = field(default_factory=MetricBlock)

    def to_payload(self) -> Dict[str, Any]:
        return jsonable(self)


@dataclass(frozen=True)
class RunManifest:
    """The record's header: identity, provenance and declared status.

    ``conda_env`` and ``precision`` are recorded and checked because they are the
    two environment facts that silently change results. The package runs under
    ``koa_project``; a number produced anywhere else is not comparable with the
    rest of the table, and ``fp32`` is what §4.1.4 states
    training used.

    ``artifact_root`` is stored as a project-relative string, not an absolute
    path, so a record stays valid when the checkout moves. The validator requires
    it to be exactly ``artifacts/<run_id>``: the pairing between the curated
    record and the heavy output it summarises is the whole audit trail, and a
    record pointing at another run's artifacts would be worse than one pointing
    nowhere.
    """

    run_id: str
    status: RunStatus = RunStatus.PENDING
    schema_version: str = RECORD_SCHEMA_VERSION
    methodology_version: str = "chapter3_v2"
    conda_env: str = "koa_project"
    precision: str = "fp32"
    artifact_root: str = ""
    created_at: Optional[str] = None
    notes: str = ""
    xray_candidate_count: int = EXPECTED_XRAY_CANDIDATES
    mri_candidate_count: int = EXPECTED_MRI_CANDIDATES
    traditional_fusion_count: int = EXPECTED_TRADITIONAL_FUSIONS
    contrastive_fusion_count: int = EXPECTED_CONTRASTIVE_FUSIONS
    rckf_fusion_count: int = EXPECTED_RCKF_FUSIONS
    paired_distribution_total: int = EXPECTED_PAIRED_TOTAL

    @property
    def is_pending(self) -> bool:
        return RunStatus(self.status) is RunStatus.PENDING

    def to_payload(self) -> Dict[str, Any]:
        """camelCase payload with ``status`` flattened to its string value.

        ``jsonable`` passes a ``str`` subclass through untouched, which serialises
        correctly today but leaves a ``RunStatus`` instance in the returned
        mapping -- so an in-memory caller comparing against ``"pending"`` would be
        relying on ``Enum`` inheritance rather than on the payload being plain
        JSON types. Converting here keeps ``to_payload`` honest.
        """

        payload = jsonable(self)
        payload["status"] = RunStatus(self.status).value
        return payload


def expected_counts() -> Dict[str, int]:
    """The structural constants, keyed by the manifest field they belong to.

    One mapping shared by the scaffold that writes the counts and the validator
    that checks them, so the two cannot drift apart -- which is the only way a
    self-consistent but wrong record could otherwise arise.
    """

    return {
        "xray_candidate_count": EXPECTED_XRAY_CANDIDATES,
        "mri_candidate_count": EXPECTED_MRI_CANDIDATES,
        "traditional_fusion_count": EXPECTED_TRADITIONAL_FUSIONS,
        "contrastive_fusion_count": EXPECTED_CONTRASTIVE_FUSIONS,
        "rckf_fusion_count": EXPECTED_RCKF_FUSIONS,
        "paired_distribution_total": EXPECTED_PAIRED_TOTAL,
    }
