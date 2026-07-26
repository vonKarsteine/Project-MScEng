# Contribution conventions

Five rules. Each one is mechanically checked, and each exists because breaking it produces something
that *looks* like it works.

## 1. Imports point strictly downward

```
core → config → data, models → fusion, stats → ensemble → deploy, api, records → training → cli
```

A module may import from a **lower** layer, or from its own package. Never from an equal or higher
one — `api` and `deploy` share layer 5, so neither may import the other.

**Every `__init__.py` stays empty.** No re-exports, ever. A populated `__init__` makes a package's
import cost unpredictable, gives one symbol several import paths, and defeats the layer check, which
reads the layer off the import line. That also means every cross-package import names its module:
`from koa_multimodal.fusion.rckf import RCKFFusion`, never `from koa_multimodal.fusion import ...`.

Nothing patches `sys.path` — the package is pip-installed. `torch.load` appears in exactly one module,
`core/checkpoint.py`, which owns the trust policy.

*Checked by* `tests/test_import_layers.py`, which walks every module's AST.

## 2. snake_case internally; camelCase only at the serialization boundary

Internal code is snake_case throughout. JSON — the HTTP API, prediction artifacts, result records — is
camelCase, because that is what the frontend and the existing dissertation records consume.

The conversion happens in exactly one place: `koa_multimodal/core/serialization.py`. Write a camelCase
key anywhere else and a rename becomes a string edit spread across five packages instead of a refactor.

## 3. Build `CandidateOutput` through `from_corn` / `from_posterior`, never by hand

The factories derive `prediction` and `uncertainty` from `probabilities`, so those three fields cannot
drift out of agreement. Constructing the dataclass directly lets them.

Which factory is not a style choice:

- `from_corn` — the model owns a `CornOrdinalHead` and has a real `[B, K-1]` conditional threshold
  chain.
- `from_posterior` — the route averages or selects posteriors (the OOF stacker, the C-MODES router)
  and owns no ordinal head, so `threshold_logits` is `None`.

Applying a CORN objective to a posterior-averaging route would read `K` thresholds off a `K`-class
probability vector: finite, plausible, meaningless. `require_threshold_logits()` is what makes that
impossible rather than merely wrong — call it instead of touching `threshold_logits` directly.

Diagnostics travel in the typed `Trace`, not in a string-keyed dict. An absent branch is `None`, never
a missing key.

## 4. Never relax `require_oof`

Anything that *trains* a selector or a stacker loads prediction artifacts with `require_oof=True`,
which rejects ordinary validation predictions outright. That gate is the methodological spine of
Chapter 3.

If a step fails because its artifacts are not out-of-fold, **produce V2 artifacts** — real per-fold
retraining on the non-held-out subjects, with per-row `subjectIds`/`foldIds` and a per-fold
`trainSubjectHash`. Do not pass `require_oof=False`, do not relabel one validation run as `"oof"`, and
do not widen the artifact schema to accept a record without fold provenance. A leaked selector still
trains, still converges, and still reports better numbers than an honest one.

Artifacts are aligned by **row order**, not by a `pairKey` join. A join quietly reorders and drops
rows; the order check fails loudly.

## 5. Never write metrics into a `pending` record

`results/<run_id>/` is the curated, dissertation-facing record tree, and `manifest.json` declares a
`status`. The validator checks both directions: a `complete` record with an empty metric block is an
interrupted run mislabelled as finished, and a **`pending` record carrying numbers is rejected**,
because that is exactly the shape transcribed or invented figures would take.

Corollaries:

- Scaffold the record tree **before** training. The trainer writes into it and will not create it.
- Reference figures belong in `results/published/`, which is read-only. **Never copy a value out of it
  into a run record.** It exists so tests can check the code's constants against the write-up — that is
  agreement with the text, not evidence of reproduction.
- Do not report new metrics until OOF provenance, untouched-test predictions and the full
  2000-iteration statistical validation have all completed.
- Record strings are scanned for placeholder words (`prefilled`, `fabricated`, `placeholder`, `todo`,
  `dummy`, `synthetic result`) outside a two-key prose allowlist. A field holds a real value or `null`,
  never a stand-in.

*Checked by* `koa records validate --all`.

---

## Language floor

Python **3.9.25**. No `match`, no `StrEnum`, no `dataclass(slots=True)`, no PEP 604 `X | Y` or PEP 585
`list[str]` in annotations evaluated at runtime. `from __future__ import annotations` at the top of
every module; `Optional[X]`, `Dict`, `List`, `Tuple` from `typing`.

There is no linter and no formatter. Match the surrounding file.

## Before opening a change

```powershell
conda run -n koa_project pytest -q
conda run -n koa_project koa check pipeline --synthetic
conda run -n koa_project koa records validate --all
conda run -n koa_project koa serve --check
```

Every new CLI entry point implements all three smoke levels — `--dry-run`, `--contract-check`, and a
resolved-contract exit for absent inputs — through `cli.common.resolve_smoke_level`. This checkout
ships no data and no weights, so those three levels are the entire proof that a change works.
