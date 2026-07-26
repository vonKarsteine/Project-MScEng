"""The single validator for a results record. One code path, every record.

What this exists to prevent is narrow and specific: a number appearing in
``results/`` that no run in this checkout produced. No training will be run here,
so every record in this tree is ``pending``, and the validator's most important
rule is the one that enforces the *absence* of results:

* ``status == "complete"`` -- every metric block must be fully populated, and both
  oracle bounds of Table 4.11 must be present. A complete record with holes is an
  interrupted run mislabelled as finished.
* ``status == "pending"`` -- the exact opposite. Every metric field must be
  ``None``. A pending record carrying numbers is rejected, because that is
  precisely how transcribed or invented figures would enter the record tree and
  acquire the authority of machine output.

Both rules are checked by the same function over the same files. Dispatching
instead on whether a ``manifest.json`` exists gives two branches that check
different things, and if the metric and oracle checks live in only one of them a
record in the other shape passes with empty metrics and zero oracle records.
Status is an explicit declared field for that reason, and the *only* thing it
changes is the direction of the metric check.

Every finding is collected. The function never returns at the first error,
because "the manifest names the wrong conda environment" and "three record files
are missing" are one fix session, not three round trips.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Tuple, Union

from koa_multimodal.config.paths import results_root
from koa_multimodal.core.serialization import camel, read_json
from koa_multimodal.records.schema import (
    EXPECTED_ORACLE_BOUNDS,
    METRIC_KEYS,
    RECORD_FILES,
    RunStatus,
    expected_counts,
)

PathLike = Union[str, Path]

#: Words that must not appear in a record. All six describe content that is
#: standing in for a result rather than being one.
#:
#: Matched on word boundaries, case-insensitively. The boundary is not pedantry:
#: a substring scan for ``provenance`` false-positives on ``foldProvenance``, a
#: legitimate and load-bearing key of the out-of-fold artifact schema, and a guard
#: that has to be worked around stops being a guard. ``todo`` would likewise hit
#: ``mastodon`` unbounded.
PLACEHOLDER_WORDS: Tuple[str, ...] = (
    "prefilled",
    "fabricated",
    "placeholder",
    "todo",
    "dummy",
    "synthetic result",
)

#: Keys whose values are descriptive prose, exempt from the lint.
#:
#: A record has to be able to *say* "these are placeholder-free pending metrics"
#: in its own notes without failing its own check. Restricting the exemption to
#: two known keys keeps the escape hatch from becoming general: an id, a path or a
#: display name is never exempt.
PROSE_KEYS: FrozenSet[str] = frozenset({"notes", "description"})

_PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:" + "|".join(word.replace(" ", r"\s+") for word in PLACEHOLDER_WORDS) + r")\b",
    re.IGNORECASE,
)

#: Scanned by the pending null-check. Wider than :data:`METRIC_KEYS` because the
#: router's switch precision is a reported result too (Table 4.10), and a pending
#: record must not carry it either.
_PENDING_SCANNED_KEYS: FrozenSet[str] = METRIC_KEYS | frozenset(
    {"switchPrecision", "switchCount"}
)

_MANIFEST_FILE = "manifest.json"
_ORACLE_FILE = "oracle_bounds.json"
_EXAMPLE_LIMIT = 8


@dataclass(frozen=True)
class ValidationReport:
    """Everything the validator established about one record directory.

    A report, not an exception: the useful output of validating a broken record
    is the full list of what is wrong with it. ``ok`` is stored rather than
    derived from ``errors`` being empty so that the report round-trips through
    JSON without the meaning of the verdict depending on a property.
    """

    run_id: str
    status: str
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    files_checked: int = 0

    def describe(self) -> str:
        """A CLI-shaped multi-line summary."""

        verdict = "ok" if self.ok else "FAILED"
        lines = [
            f"record {self.run_id}: {verdict} "
            f"(status={self.status}, {self.files_checked}/{len(RECORD_FILES)} files read, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings)"
        ]
        for message in self.errors:
            lines.append(f"  error   {message}")
        for message in self.warnings:
            lines.append(f"  warning {message}")
        return "\n".join(lines)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "files_checked": self.files_checked,
        }


# ------------------------------------------------------------------- traversal


def _iter_nodes(node: Any, path: str = "") -> Iterator[Tuple[str, Any]]:
    """Every node of a parsed JSON document, with a dotted path to it."""

    yield path, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _iter_nodes(value, f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_nodes(value, f"{path}[{index}]")


def _leaf_key(path: str) -> str:
    """The final key of a dotted path, with any list indices stripped."""

    tail = path.rsplit(".", 1)[-1]
    return tail.split("[", 1)[0]


def _iter_metric_blocks(document: Any) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Every serialised :class:`~koa_multimodal.records.schema.MetricBlock`.

    A block is recognised by carrying *all four* metric keys, which is what
    ``MetricBlock.to_payload`` always emits. Requiring all four is what keeps a
    confidence-interval mapping -- three metric keys, no ``sampleCount`` -- from
    being mistaken for a block and then failed for the missing support count.
    """

    for path, node in _iter_nodes(document):
        if isinstance(node, dict) and METRIC_KEYS.issubset(node.keys()):
            yield path, node


def _iter_metric_fields(document: Any) -> Iterator[Tuple[str, str, Any]]:
    """Every metric-named field anywhere in the document, block or not.

    Deliberately broader than :func:`_iter_metric_blocks`: the pending check has
    to catch a stray ``"accuracy": 0.872`` sitting outside any well-formed block,
    which is exactly the shape a hand-edited number would take.
    """

    for path, node in _iter_nodes(document):
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if key in _PENDING_SCANNED_KEYS:
                yield (f"{path}.{key}" if path else str(key)), key, value


def _iter_strings(document: Any) -> Iterator[Tuple[str, str]]:
    """Every string *value* in the document, with its path. Keys are not scanned."""

    for path, node in _iter_nodes(document):
        if isinstance(node, str):
            yield path, node


# --------------------------------------------------------------------- checks


def _load_records(
    directory: Path, errors: List[str]
) -> Tuple[Dict[str, Any], int]:
    """Read every required record file, reporting each failure separately."""

    documents: Dict[str, Any] = {}
    for name in RECORD_FILES:
        source = directory / name
        if not source.is_file():
            errors.append(f"missing required record file: {name}")
            continue
        try:
            documents[name] = read_json(source)
        except json.JSONDecodeError as exc:
            errors.append(f"{name}: not valid JSON ({exc})")
        except OSError as exc:
            errors.append(f"{name}: could not be read ({exc})")
    return documents, len(documents)


def _check_manifest(
    run_id: str, manifest: Dict[str, Any], errors: List[str]
) -> str:
    """Identity, environment and structural counts. Returns the declared status."""

    declared_id = manifest.get("runId")
    if declared_id != run_id:
        errors.append(
            f"manifest.json: runId is {declared_id!r} but the directory is named "
            f"{run_id!r}; a record that does not know its own name cannot be joined "
            "to its artifacts"
        )

    for key, expected in (("condaEnv", "koa_project"), ("precision", "fp32")):
        actual = manifest.get(key)
        if actual != expected:
            errors.append(f"manifest.json: {key} is {actual!r}, expected {expected!r}")

    expected_root = f"artifacts/{run_id}"
    artifact_root = manifest.get("artifactRoot")
    if artifact_root != expected_root:
        errors.append(
            f"manifest.json: artifactRoot is {artifact_root!r}, expected "
            f"{expected_root!r} (project-relative, and pointing at this run's own "
            "artifacts)"
        )

    for name, expected_value in expected_counts().items():
        key = camel(name)
        actual = manifest.get(key)
        if actual != expected_value:
            errors.append(
                f"manifest.json: {key} is {actual!r}, expected {expected_value} "
                "from the dissertation's Chapter 4"
            )

    status = manifest.get("status")
    if status not in tuple(member.value for member in RunStatus):
        errors.append(
            f"manifest.json: status is {status!r}, expected one of "
            f"{[member.value for member in RunStatus]}"
        )
        return str(status)
    return str(status)


def _check_metrics_against_status(
    status: str, documents: Dict[str, Any], errors: List[str]
) -> None:
    """The status-conditional rule, in both directions."""

    if status == RunStatus.COMPLETE.value:
        for name, document in documents.items():
            for path, block in _iter_metric_blocks(document):
                empty = sorted(key for key in METRIC_KEYS if block.get(key) is None)
                if empty:
                    errors.append(
                        f"{name}: status is 'complete' but the metric block at "
                        f"{path or '<root>'} leaves {empty} unset"
                    )
        _check_oracle_bounds(documents, errors)
        return

    # Everything else -- ``pending``, and any unrecognised status the manifest
    # check has already reported -- takes the null rule. Falling through to the
    # stricter of the two matters: skipping the check for an unknown status would
    # make ``"status": "final"`` a way to carry numbers past the validator.
    for name, document in documents.items():
        offenders = [
            f"{path}={value!r}"
            for path, _key, value in _iter_metric_fields(document)
            if value is not None
        ]
        if offenders:
            errors.append(
                f"{name}: status is {status!r} but these metric fields carry values: "
                f"{offenders[:_EXAMPLE_LIMIT]}"
                + (f" (+{len(offenders) - _EXAMPLE_LIMIT} more)" if len(offenders) > _EXAMPLE_LIMIT else "")
                + ". No run has produced them, so they cannot be recorded here; "
                "reference figures belong in results/published/."
            )


def _check_oracle_bounds(documents: Dict[str, Any], errors: List[str]) -> None:
    """Table 4.11 reports two ceilings, and a complete record must carry both."""

    document = documents.get(_ORACLE_FILE)
    if document is None:
        return  # already reported as a missing or unparseable file
    bounds = document.get("bounds") if isinstance(document, dict) else None
    if not isinstance(bounds, list):
        errors.append(f"{_ORACLE_FILE}: expected a 'bounds' array of oracle records")
        return
    pool_ids = [entry.get("poolId") for entry in bounds if isinstance(entry, dict)]
    if len(pool_ids) != EXPECTED_ORACLE_BOUNDS:
        errors.append(
            f"{_ORACLE_FILE}: status is 'complete' but {len(pool_ids)} oracle bounds "
            f"are recorded, expected {EXPECTED_ORACLE_BOUNDS} (Table 4.11: the "
            "X-ray-only and bimodal four-model selection ceilings)"
        )


def _check_placeholders(documents: Dict[str, Any], errors: List[str]) -> None:
    """Reject stand-in text in any string value outside the prose allowlist."""

    for name, document in documents.items():
        for path, text in _iter_strings(document):
            if _leaf_key(path) in PROSE_KEYS:
                continue
            match = _PLACEHOLDER_PATTERN.search(text)
            if match is not None:
                errors.append(
                    f"{name}: {path or '<root>'} contains the blocked word "
                    f"{match.group(0)!r}; a record field must hold a real value or "
                    "be null, never a stand-in"
                )


# ---------------------------------------------------------------- entry points


def validate_record_dir(directory: PathLike) -> ValidationReport:
    """Validate one record directory, collecting every finding.

    ``run_id`` is taken from the directory name, which is what makes the
    ``runId`` check meaningful -- the record is compared against where it
    actually lives, not against a value it supplies twice.
    """

    path = Path(directory)
    run_id = path.name
    errors: List[str] = []
    warnings: List[str] = []

    if not path.is_dir():
        return ValidationReport(
            run_id=run_id,
            status="unknown",
            ok=False,
            errors=[
                f"record directory does not exist: {path}. Scaffold it before "
                "training with koa_multimodal.records.scaffold.scaffold_run; the "
                "trainer writes into this directory and will not create it."
            ],
            warnings=warnings,
            files_checked=0,
        )

    documents, files_checked = _load_records(path, errors)

    manifest = documents.get(_MANIFEST_FILE)
    if isinstance(manifest, dict):
        status = _check_manifest(run_id, manifest, errors)
    else:
        if _MANIFEST_FILE in documents:
            errors.append(f"{_MANIFEST_FILE}: expected a JSON object at the top level")
        status = "unknown"

    unexpected = sorted(
        entry.name
        for entry in path.iterdir()
        if entry.is_file() and entry.name not in RECORD_FILES
    )
    if unexpected:
        warnings.append(
            f"files present that are not part of the record schema: {unexpected}"
        )

    _check_metrics_against_status(status, documents, errors)
    _check_placeholders(documents, errors)

    return ValidationReport(
        run_id=run_id,
        status=status,
        ok=not errors,
        errors=errors,
        warnings=warnings,
        files_checked=files_checked,
    )


def validate_run(run_id: str, *, root: Optional[PathLike] = None) -> ValidationReport:
    """Validate ``results/<run_id>/``.

    ``root`` overrides the results directory for tests; by default the location
    comes from :func:`koa_multimodal.config.paths.results_root`, so validation
    does not depend on the caller's working directory.
    """

    directory = Path(root) / run_id if root is not None else results_root(run_id)
    return validate_record_dir(directory)
