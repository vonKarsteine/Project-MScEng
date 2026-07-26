"""Create an honest, empty record directory before a run starts.

``results/<run_id>/`` has to exist *before* training, because the trainer writes
into it and does not create it. Left to the trainer, the missing directory
surfaces as a hard failure at the end of a long stage, after the expensive part
has already run; scaffolding first turns it into a one-second precondition.

What gets written is a complete record with every metric field ``null``. That is
the point, not a limitation:

* The structural facts are real -- the candidate and fusion ids come from
  :mod:`koa_multimodal.models.catalog` and
  :data:`koa_multimodal.core.ids.FUSION_ROUTES`, the counts from Chapter 4, the
  hyperparameters from the resolved :class:`~koa_multimodal.config.schema.Config`.
  These are *inputs*: they describe what a run would do, and they are knowable
  before it does it.
* The metrics are ``null``, and :mod:`koa_multimodal.records.validate` requires
  them to stay that way while the status is ``pending``. A scaffolded record
  therefore cannot quietly acquire numbers; filling one in by hand fails
  validation.

Where a pool's membership is stated in the dissertation only by family rather
than by model -- the bimodal ensemble and oracle pools -- the id list is left empty and
the wording is transcribed into ``description`` instead. Guessing four ids that
happen to satisfy the count would put a claim into the record tree that no source
supports, which is the failure this whole layer exists to prevent.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from koa_multimodal.config.loader import config_to_mapping
from koa_multimodal.config.paths import results_root
from koa_multimodal.config.schema import Config
from koa_multimodal.core.errors import KoaError
from koa_multimodal.core.ids import FUSION_ROUTES, fusion_candidate_id
from koa_multimodal.core.serialization import write_json
from koa_multimodal.models.catalog import (
    MRI_CANDIDATES,
    XRAY_CANDIDATES,
    candidate_spec,
)
from koa_multimodal.records.schema import (
    EXPECTED_CONTRASTIVE_FUSIONS,
    EXPECTED_RCKF_FUSIONS,
    EXPECTED_TRADITIONAL_FUSIONS,
    ORACLE_POOL_IDS,
    RECORD_FILES,
    RECORD_SCHEMA_VERSION,
    CandidateRecord,
    MetricBlock,
    OracleRecord,
    RunManifest,
    RunStatus,
    expected_counts,
)

PathLike = Union[str, Path]

#: Human labels for the five routes, used to compose fusion display names.
_ROUTE_LABELS: Dict[str, str] = {
    "late_concat": "Late concatenation",
    "gated": "Gated",
    "cross_attention": "Cross-attention",
    "contrastive": "Contrastive",
    "rckf": "RCKF",
}

#: Table 4.7. All three traditional routes are reported over one pairing, so the
#: comparison between them isolates the fusion operator.
_TRADITIONAL_ROUTES: Tuple[str, ...] = ("late_concat", "gated", "cross_attention")
_TRADITIONAL_PAIRING: Tuple[str, str] = ("xray_convnext_v2", "mri_simsiam_ssl")

#: Tables 4.8 and 4.9 report the same 2 x 2 grid of branches, which is what makes
#: the contrastive-to-RCKF comparison a controlled one.
_BIMODAL_XRAY_BRANCHES: Tuple[str, ...] = ("xray_convnext_v2", "xray_maxvit")
_BIMODAL_MRI_BRANCHES: Tuple[str, ...] = ("mri_simsiam_ssl", "mri_swin3d")

#: Table 4.10 names this pool member by member; DeiT, the weakest row of Table
#: 4.5, is not in it.
_XRAY_POOL: Tuple[str, ...] = (
    "xray_convnext_v2",
    "xray_maxvit",
    "xray_efficientnet_v2",
    "xray_swin_t",
)

_BIMODAL_POOL_DESCRIPTION = (
    "Table 4.10 states this pool by family -- the gated fusion route "
    "together with the contrastive and RCKF candidates -- rather than by model. "
    "The id list is left empty on purpose: Table 4.11 calls the matching oracle a "
    "four-model bound, and no member-by-member reading of the family wording is "
    "determined by the text. A run resolves it from its pool configuration and "
    "writes the ids it actually used."
)


def _timestamp() -> str:
    """UTC, second resolution, ``Z``-suffixed."""

    stamp = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    return stamp.isoformat() + "Z"


def _header(run_id: str) -> Dict[str, Any]:
    """The three fields every record file repeats.

    Repeated deliberately: a record file lifted out of its directory still says
    which run it belongs to and whether its numbers mean anything.
    """

    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "status": RunStatus.PENDING.value,
    }


def _manifest(run_id: str, config: Config) -> Dict[str, Any]:
    manifest = RunManifest(
        run_id=run_id,
        status=RunStatus.PENDING,
        schema_version=RECORD_SCHEMA_VERSION,
        methodology_version=config.project.methodology_version,
        conda_env=config.training.environment,
        precision=config.training.precision,
        artifact_root="artifacts/{0}".format(run_id),
        created_at=_timestamp(),
        notes=(
            "Scaffolded, not run. Every metric field in this record is null and "
            "must stay null while status is pending; the validator rejects a "
            "pending record that carries numbers. The figures reported in "
            "Chapter 4 of the dissertation were produced elsewhere, on different "
            "hardware, and are kept, clearly labelled, under "
            "results/published/. They are reference values for tests and must "
            "never be copied into this tree."
        ),
        **expected_counts()
    )
    return manifest.to_payload()


def _candidates(run_id: str) -> Dict[str, Any]:
    """The nine unimodal rows of Tables 4.5 and 4.6, in published order.

    ``display_name`` is the catalogue's, and matches the row label in the thesis
    tables verbatim -- that string is the join between this record and the
    published transcription.
    """

    records = [
        CandidateRecord(
            candidate_id=spec.candidate_id,
            modality=spec.modality,
            display_name=spec.display_name,
            validation=MetricBlock.pending(),
            test=MetricBlock.pending(),
            checkpoint_hash=None,
        )
        for spec in XRAY_CANDIDATES + MRI_CANDIDATES
    ]
    return {
        **_header(run_id),
        "description": (
            "Unimodal candidates, in the order of Tables 4.5 (X-ray) "
            "and 4.6 (MRI). Selection is on validation QWK, so both splits are "
            "recorded separately and only the test block is reportable."
        ),
        "candidates": [asdict(record) for record in records],
    }


def _fusion_row(route: str, xray_id: str, mri_id: str, family: str) -> Dict[str, Any]:
    candidate_id = fusion_candidate_id(route, xray_id, mri_id)
    display_name = "{0} ({1} + {2})".format(
        _ROUTE_LABELS[route],
        candidate_spec(xray_id).display_name,
        candidate_spec(mri_id).display_name,
    )
    record = CandidateRecord(
        candidate_id=candidate_id,
        modality="fusion",
        display_name=display_name,
        validation=MetricBlock.pending(),
        test=MetricBlock.pending(),
        checkpoint_hash=None,
    )
    return {
        **asdict(record),
        "route": route,
        "family": family,
        "xray_candidate_id": xray_id,
        "mri_candidate_id": mri_id,
    }


def _fusion(run_id: str) -> Dict[str, Any]:
    """The eleven fusion rows of Tables 4.7, 4.8 and 4.9.

    Composed from the catalogue rather than listed, so the ids here and the ids a
    run trains come from one source. The count assertions below are the tripwire:
    if a route or a branch is added without updating Chapter 4's constants, the
    scaffold fails here rather than producing a record the validator will reject
    later for reasons that point at the wrong file.
    """

    traditional = [
        _fusion_row(route, _TRADITIONAL_PAIRING[0], _TRADITIONAL_PAIRING[1], "traditional")
        for route in _TRADITIONAL_ROUTES
    ]
    contrastive = [
        _fusion_row("contrastive", xray_id, mri_id, "contrastive")
        for xray_id in _BIMODAL_XRAY_BRANCHES
        for mri_id in _BIMODAL_MRI_BRANCHES
    ]
    rckf = [
        _fusion_row("rckf", xray_id, mri_id, "rckf")
        for xray_id in _BIMODAL_XRAY_BRANCHES
        for mri_id in _BIMODAL_MRI_BRANCHES
    ]

    for family, rows, expected in (
        ("traditional", traditional, EXPECTED_TRADITIONAL_FUSIONS),
        ("contrastive", contrastive, EXPECTED_CONTRASTIVE_FUSIONS),
        ("rckf", rckf, EXPECTED_RCKF_FUSIONS),
    ):
        if len(rows) != expected:
            raise KoaError(
                "{0} fusion family builds {1} routes but Chapter 4 reports {2}; "
                "the catalogue and the expected counts in records.schema have "
                "diverged.".format(family, len(rows), expected)
            )
    unknown = sorted({row["route"] for row in traditional + contrastive + rckf} - set(FUSION_ROUTES))
    if unknown:
        raise KoaError("fusion routes not in core.ids.FUSION_ROUTES: {0}".format(unknown))

    return {
        **_header(run_id),
        "description": (
            "Fusion routes, in the order of Tables 4.7 (traditional), "
            "4.8 (contrastive) and 4.9 (RCKF). The traditional routes share one "
            "branch pairing so that the comparison isolates the fusion operator; "
            "the contrastive and RCKF families span the same two-by-two grid of "
            "branches so that the comparison between them is controlled."
        ),
        "routes": traditional + contrastive + rckf,
    }


def _ensemble(run_id: str) -> Dict[str, Any]:
    """The four rows of Table 4.10.

    ``switch_precision`` and ``switch_count`` sit beside the metrics because
    neither is interpretable alone: the traditional-ensemble rows report a switch
    precision of 0.000 for the simple reason that they never switch, and only the
    count distinguishes that from a router that switches often and badly.
    """

    def profile(
        profile_id: str, strategy: str, pool: List[str], description: str
    ) -> Dict[str, Any]:
        return {
            "profile_id": profile_id,
            "strategy": strategy,
            "description": description,
            "pool_candidate_ids": pool,
            "test": asdict(MetricBlock.pending()),
            "switch_precision": None,
            "switch_count": None,
        }

    xray_pool_description = (
        "Table 4.10: the X-ray pool of ConvNeXt V2, MaxViT, "
        "EfficientNet V2 and Swin-T."
    )
    return {
        **_header(run_id),
        "description": (
            "Ensemble and routing profiles, in the order of Table 4.10. "
            "Switch precision is the ordinal definition of "
            "koa_multimodal.stats.metrics.switch_precision -- a switch counts as "
            "correct when it lands closer to the truth than the default route "
            "would have -- and is meaningless without switchCount beside it."
        ),
        "profiles": [
            profile(
                "xray_traditional_ensemble",
                "posterior_averaging",
                list(_XRAY_POOL),
                xray_pool_description,
            ),
            profile(
                "xray_cmodes",
                "cmodes_routing",
                list(_XRAY_POOL),
                xray_pool_description,
            ),
            profile(
                "multimodal_traditional_ensemble",
                "posterior_averaging",
                [],
                _BIMODAL_POOL_DESCRIPTION,
            ),
            profile(
                "multimodal_cmodes",
                "cmodes_routing",
                [],
                _BIMODAL_POOL_DESCRIPTION,
            ),
        ],
    }


def _oracle_bounds(run_id: str) -> Dict[str, Any]:
    """The two selection ceilings of Table 4.11.

    Not achievable and not meant to be: the oracle picks the best pool member per
    sample using the true label, so the gap between it and the C-MODES row of
    Table 4.10 is the headroom a better router could still recover. Both bounds
    are required by the validator once a record is complete.
    """

    bounds = [
        OracleRecord(
            pool_id=ORACLE_POOL_IDS[0],
            candidate_ids=list(_XRAY_POOL),
            metrics=MetricBlock.pending(),
        ),
        OracleRecord(
            pool_id=ORACLE_POOL_IDS[1],
            candidate_ids=[],
            metrics=MetricBlock.pending(),
        ),
    ]
    return {
        **_header(run_id),
        "description": (
            "Per-sample selection ceilings over a fixed pool, Table "
            "4.11. " + _BIMODAL_POOL_DESCRIPTION
        ),
        "bounds": [asdict(record) for record in bounds],
    }


def _statistical_validation(run_id: str, config: Config) -> Dict[str, Any]:
    """Resampling parameters now; intervals once there is something to resample.

    The parameters are inputs and are recorded in full. ``intervals`` and
    ``paired_comparisons`` are empty because no predictions exist to resample --
    an empty array is the accurate statement, and it is what a reader should see
    instead of interval bounds with no run behind them.
    """

    statistics = config.statistics
    return {
        **_header(run_id),
        "description": (
            "Subject-clustered bootstrap, dissertation §3.5. Resampling is "
            "over subjects rather than samples because a patient's two knees are "
            "not independent observations."
        ),
        "bootstrap_iterations": statistics.bootstrap_iterations,
        "bootstrap_seed": statistics.bootstrap_seed,
        "bootstrap_alpha": statistics.bootstrap_alpha,
        "cluster_by_subject": statistics.cluster_by_subject,
        "group_stratum_rule": (
            "each subject is assigned to the stratum of its highest KL grade, "
            "because a patient's two knees routinely differ and no exact "
            "label-pure clustering of subjects exists"
        ),
        "intervals": [],
        "paired_comparisons": [],
    }


def _qat_deployment(run_id: str) -> Dict[str, Any]:
    """Student-versus-teacher accounting and the export contract.

    ``contract_fields`` is ordered, and the order is the contract: the same
    sequence has to hold across the deployment payload, the ONNX output names and
    this record, or a consumer reading position 3 gets the Kalman gain where it
    expected the measurement variance.
    """

    return {
        **_header(run_id),
        "description": (
            "Quantisation-aware distillation of the deployment route, "
            "Table 4.15. The precision profile is mixed by design: only the X-ray "
            "encoders and the selector are fake-quantised, while the MRI and RCKF "
            "branch and the ordinal head are not."
        ),
        "contract_version": "qat-v2",
        "contract_fields": [
            "probabilities",
            "posteriorLatent",
            "measurementVariance",
            "kalmanGain",
            "selectorScores",
            "selectedRoute",
            "missingModalityFallback",
        ],
        "student": asdict(MetricBlock.pending()),
        "teacher": asdict(MetricBlock.pending()),
        "delta": {"accuracy": None, "qwk": None, "kl1_recall": None},
        "export": {
            "format": "onnx",
            "exported": False,
            "reason": (
                "no run has produced a student to export, and the deployment "
                "optional extra is not installed in this environment"
            ),
        },
    }


def _training_parameters(run_id: str, config: Config) -> Dict[str, Any]:
    """The fully resolved configuration.

    Legitimately populated in a pending record, and the one file here that is:
    these are the settings a run *would* use, not measurements it produced. None
    of the keys is metric-named, so the validator's null rule does not touch them.
    Recording the resolved values rather than the TOML text captures environment
    overrides, which is what makes the record reproducible.
    """

    return {
        **_header(run_id),
        "description": (
            "The resolved configuration this run would train under, including any "
            "KOA_<SECTION>__<KEY> environment overrides applied at load time. "
            "Inputs, not results."
        ),
        "configuration": config_to_mapping(config),
    }


def scaffold_run(
    run_id: str,
    *,
    config: Config,
    force: bool = False,
    root: Optional[PathLike] = None,
) -> Path:
    """Create ``results/<run_id>/`` as a complete, honest pending record.

    Returns the directory. Every file in :data:`RECORD_FILES` is written, so the
    result passes :func:`koa_multimodal.records.validate.validate_run`
    immediately -- a scaffold that produced a record needing manual repair would
    just move the problem.

    Refuses to write into a directory that already holds anything, because the
    files it writes are exactly the files a finished run overwrites, and an
    accidental re-scaffold would erase real results with nulls. ``force=True`` is
    the deliberate override. An existing *empty* directory is not a record and is
    filled in normally.

    ``root`` overrides the results location for tests.
    """

    directory = Path(root) / run_id if root is not None else results_root(run_id)
    if directory.exists():
        occupants = sorted(entry.name for entry in directory.iterdir())
        if occupants and not force:
            raise KoaError(
                "{0} already exists and is not empty ({1}). Scaffolding would "
                "overwrite every record file with nulls; pass force=True if that "
                "is what you mean.".format(directory, occupants[:8])
            )
    directory.mkdir(parents=True, exist_ok=True)

    documents: Dict[str, Dict[str, Any]] = {
        "manifest.json": _manifest(run_id, config),
        "candidates.json": _candidates(run_id),
        "fusion.json": _fusion(run_id),
        "ensemble.json": _ensemble(run_id),
        "oracle_bounds.json": _oracle_bounds(run_id),
        "statistical_validation.json": _statistical_validation(run_id, config),
        "qat_deployment.json": _qat_deployment(run_id),
        "training_parameters.json": _training_parameters(run_id, config),
    }
    missing = sorted(set(RECORD_FILES) - set(documents))
    if missing:
        raise KoaError(
            "scaffold_run builds no content for {0}; RECORD_FILES and this "
            "function have diverged.".format(missing)
        )

    for name in RECORD_FILES:
        write_json(documents[name], directory / name)
    return directory
