# Runbook

The six dissertation sections, stage by stage: what each needs, what it produces, and why it has to
come after the one before it.

Each section is a declarative manifest under `pipelines/`. `koa pipeline run <name>` **resolves and
describes** every step against the config — batch sizes, accumulation, objective weights, candidate
ids — without executing anything. That is deliberate: a stage's plan is data a reader can inspect,
rather than control flow they have to trace through wrapper scripts.

```powershell
conda run -n koa_project koa pipeline list
conda run -n koa_project koa pipeline run 03_fusion --dry-run
```

## Before anything: read this

**No stage in this document has been executed in this checkout, and none can be.** `data/` is empty
(the OAI subset is not redistributable), so every step that touches real data stops at index load. The
run book in `README.md` is what *is* verifiable here. Treat this document as the procedure for a
checkout that has the data, and as the map that makes the plan output legible in one that does not.

Two CLI targets named by the manifests do not exist yet in this checkout, and `koa pipeline run` does
not fail on them because it only describes steps:

- `command = "ensemble", operation = "oof"` (pipeline 04) — `koa ensemble` offers `stack`, `oracle`
  and `cmodes`. There is no `oof` target, so out-of-fold artifact *generation* has no entry point.
- `command = "stats", operation = "validate"` (pipeline 05, final step) — there is no `koa stats`
  command. The statistics library (`koa_multimodal.stats`) is complete and tested; only the CLI
  surface is missing.

Everything downstream of pipeline 04 therefore has a documented plan and a tested library behind it,
and no way to be driven from the command line yet.

## The three smoke levels

Every command implements the same three, so the whole pipeline can be exercised with no data, no
weights and no GPU:

| Level | Flag | Behaviour |
|---|---|---|
| Plan | `--dry-run` | Print the fully resolved plan, exit 0, touch nothing. |
| Contract | `--contract-check` | Real work on synthetic tensors: build the model, forward one sample, assert shapes. Writes nothing. |
| Resolved contract | *(neither flag, required inputs absent)* | Print the resolved contract and exit 0. |

`koa_multimodal.cli.common.resolve_smoke_level` is the single implementation. Shared flags on every
subcommand: `--config`, `--run-id`, `--dry-run`, `--contract-check`.

---

## 01_data — Data distribution and preprocessing

*§4.1* · `pipelines/01_data.toml`

**Needs.** The OAI subset under `data/`, and the paired index at
`data/training/xray/index/pairs_00m.csv`: eight columns —
`subject_id, laterality, split, grade, xray_relpath, t2_relpath, r2_relpath, pair_key`.

That one CSV is the source of truth for every split, label and relative path. `pair_key` is the global
primary key joining OOF artifacts, C-MODES alignment and deployment records; `subject_id` is the
grouping key for every leakage control and for the clustered bootstrap; `laterality` is pure
passthrough that nothing branches on.

**Steps.**

```powershell
conda run -n koa_project koa check data --root data --allow-empty
conda run -n koa_project koa records scaffold --run-id run_001
```

`koa check data` audits layout, per-split and per-grade counts, **subject-level leakage across
splits**, and whether every referenced file is on disk. It is read-only and never raises for a
finding — an absent dataset is a legitimate state of a distributed working copy, so it comes back as a
report with `verdict: "dataset absent -- expected in this checkout"`.

`koa records scaffold` creates `results/<run_id>/` as an honest **pending** record: eight JSON files
with every metric field `null`. This must happen **before** training — the trainer writes into that
directory and will not create it.

**Produces.** A validated index and a pending record tree.

**Preprocessing contract** (implemented in `data/xray.py`, `data/mri.py`, `data/augment.py`):

- X-ray → grayscale → knee-centre square crop → resize (bicubic, via a deliberate uint8 round trip) →
  CLAHE → `(1, 384, 384)` float32 in `[0, 1]`.
- MRI → two NIfTI volumes → T2 clipped to `[0, 110]` and divided by 110, R2 clipped to `[0, 1]` →
  `(2, 32, 384, 384)` float32, channel order **(T2, R2)**. Clipping happens *before* the resample, so
  an out-of-range voxel cannot bleed into its trilinear neighbours first.
- Augmentation runs only when `split == "train"` and is a bit-exact no-op otherwise.
- **ImageNet-gray normalisation `(x - 0.449) / 0.226` runs after augmentation**, never before. The
  augmenter finishes by clipping to `[0, 1]`, which is correct for image data and destructive for
  standardised data. MRI is never ImageNet-normalised — the physical scale is the signal.

---

## 02_unimodal — Single-modal candidate selection

*§4.2, Tables 4.5–4.6* · `pipelines/02_unimodal.toml`

**Needs.** 01 complete.

**Steps.** Five X-ray candidates, then SimSiam pretraining, then four MRI candidates:

```powershell
conda run -n koa_project koa train xray --xray-candidate-id xray_convnext_v2 --run-id run_001
#   ... xray_maxvit, xray_swin_t, xray_efficientnet_v2, xray_deit
conda run -n koa_project koa train mri_ssl --mri-candidate-id mri_simsiam_ssl --run-id run_001
conda run -n koa_project koa train mri --mri-candidate-id mri_simsiam_ssl --run-id run_001
#   ... mri_swin3d, mri_r3d18_attn, mri_r3d18
```

**Ordering matters.** `mri_ssl` comes before `mri`: KL labels are native to radiographs, so supervising
the volumetric branch with them directly imports label misalignment. SimSiam learns the volumetric
representation first. `SimSiamMriCandidate` uses LayerNorm rather than the canonical BatchNorm1d
because MRI trains at batch size 1 — do not "fix" that back.

**Selection is on validation QWK**, not accuracy (`[training] checkpoint_metric = "val_qwk"`).

**Produces.** Nine branch checkpoints under `artifacts/<run_id>/`, and per-candidate metric blocks in
`results/<run_id>/candidates.json`.

**Batching.** X-ray runs at batch 4 with no accumulation. MRI runs at physical batch **1** with
gradient accumulation 4 — MRI decode and 3-D resize happen per sample per epoch on CPU, and that cost,
not a modelling choice, sets the physical batch. See `dissertation_alignment.md` §2.

---

## 03_fusion — Multimodal fusion candidates

*§4.3, Tables 4.7–4.9* · `pipelines/03_fusion.toml`

**Needs.** 02 complete. Branch checkpoints **warm-start the encoders only** —
`fusion/assembly.py::build_fusion` grafts `XrayCandidate.backbone` and `MriCandidate.encoder` into the
route and **discards both branch CORN heads**, because the fusion route owns a new head over a new
latent space and a transplanted head would be scored on features it never saw. The graft loads
*strictly*: a checkpoint supplying a handful of coincidentally-named tensors is rejected rather than
producing a mostly-random encoder that reports success.

**Steps.** Nine routes, in the order that makes the argument:

```powershell
conda run -n koa_project koa train late_concat      --xray-candidate-id xray_convnext_v2 --mri-candidate-id mri_simsiam_ssl
conda run -n koa_project koa train gated            ...
conda run -n koa_project koa train cross_attention  ...
conda run -n koa_project koa train contrastive      ...   # x2 pairings
conda run -n koa_project koa train rckf             ...   # x4 pairings
```

The sequence is the evidence. Traditional fusion barely beats the X-ray-only model, and cross
attention — the most expressive of the three — is *worse* than gating, because a stronger interaction
module amplifies heterogeneous cross-modal noise as readily as signal. Contrastive alignment improves
on that by implicitly filtering the uncorrelated component, but applies the same pull to every sample.
RCKF adds what neither has: a per-sample estimate of how far *this* measurement can be trusted.

**Produces.** Eleven fusion checkpoints (3 traditional + 4 contrastive + 4 RCKF) and
`results/<run_id>/fusion.json`.

**Constraint.** `[training] fusion_batch_size` must be ≥ 2. `bidirectional_info_nce` returns a
gradient-connected zero below that, and the contrastive route would silently degrade to plain CORN;
`TrainingConfig.validate` raises rather than letting it.

---

## 04_oof — Out-of-fold meta-feature construction

*§3.3.1* · `pipelines/04_oof.toml`

**Needs.** 03 complete. This is the methodological spine and the step most easily faked.

**What it must do.** For each of the two pools (`configs/pools/xray.toml`, `configs/pools/fusion.toml`),
produce a `prediction-artifact-v2` per member containing, for every row, the label, prediction,
five posteriors, `pairKey`, `subjectId` and `foldId`; and for every fold, a `foldProvenance` record
carrying `checkpointHash` and `trainSubjectHash`.

Each fold requires a **genuine retraining on the non-held-out subjects** — not a relabelling of one
validation run. `trainSubjectHash` is sha256 over the sorted, de-duplicated training subject ids and is
reproducible from `OOFFoldManifest`, so a reader can verify cryptographically that fold-*k* predictions
came from a model that never saw fold *k*.

Folds are subject-grouped and grade-stratified (`data/folds.py::grouped_stratified_fold_ids`, 3 folds,
seed 42): a patient contributes two knees, and splitting them across folds leaks.

**Produces.** `artifacts/<run_id>/oof/<candidate_id>.json` for all eight pool members — the paths the
two pool files already name.

**Status.** No CLI target implements this yet (see *Before anything*). The artifact reader, validator
and fold machinery are complete and tested; the generator is not wired to a command.

**The gate.** Everything in 05 loads artifacts with `require_oof=True`, which rejects ordinary
validation predictions outright. **Never relax it** — produce V2 artifacts instead. Artifacts are
aligned by **row order, not by a pairKey join**; `validate_aligned` errors on any mismatch, because a
join would quietly reorder or drop rows where the order check fails loudly.

---

## 05_ensemble — Ensemble, oracle gap, statistical validation

*§4.4–4.5, Tables 4.10–4.13* · `pipelines/05_ensemble.toml`

**Needs.** 04 complete: OOF artifacts for both pools.

**Steps, in order.**

```powershell
conda run -n koa_project koa ensemble stack  --pool xray   --run-id run_001 --output artifacts/run_001/ensemble/xray_stack.json
conda run -n koa_project koa ensemble cmodes --pool xray   --run-id run_001 --output artifacts/run_001/ensemble/xray_cmodes.json
conda run -n koa_project koa ensemble stack  --pool fusion --run-id run_001 --output artifacts/run_001/ensemble/fusion_stack.json
conda run -n koa_project koa ensemble cmodes --pool fusion --run-id run_001 --output artifacts/run_001/ensemble/fusion_cmodes.json
conda run -n koa_project koa ensemble oracle --pool xray   --run-id run_001 --output artifacts/run_001/ensemble/xray_oracle.json
conda run -n koa_project koa ensemble oracle --pool fusion --run-id run_001 --output artifacts/run_001/ensemble/fusion_oracle.json
# statistical validation: library complete, no CLI target yet
```

`stack` is the **baseline**: a static weighted posterior average that never switches route, which is
what C-MODES has to beat. `oracle` is the **ceiling**, and it is not a result — it is the honest
measure of remaining headroom, and the reason for routing rather than for bigger models.

`cmodes` does two things in one step: it trains the selector under the simulated federated loop, and
then **calibrates `tau_s` and `c_switch` on the out-of-fold predictions** and writes them into the
selector checkpoint's `hyperparameters`. Calibrating on validation or test would leak the evaluation
set into the routing policy. Only the sum is identifiable, so the calibration sweep deduplicates on
`threshold = tau_s + c_switch`, and the objective is OOF QWK with switch precision as a feasibility
constraint rather than the target (a policy that never switches trivially maximises switch precision).

**Statistical validation** (`koa_multimodal.stats`) is a subject-clustered stratified bootstrap at
B = 2000, alpha = 0.05, seed 7, followed by *planned* paired comparisons of the final routed model
against two pre-declared baselines with Holm–Bonferroni correction. Two departures from a plain
bootstrap, both necessary:

- **Stratified by KL grade**, because KL4 is 22 of 673 test pairs and a uniform resample would draw
  pseudo-cohorts containing none of it.
- **Clustered by subject**, because a patient contributes two knees and treating them as independent
  understates the interval width.

Each subject is assigned the stratum of its **highest** KL grade; the rule travels in every payload as
`groupStratumRule`. Resamples where a metric is undefined are *dropped*, not counted as 1.0, and the
surviving count is reported as `validDraws`. The p-value uses the add-one form `(1 + count) / (B + 1)`
per tail, because a bootstrap cannot resolve below `1 / (B + 1)` and the naive form reports exactly
0.000 — which Holm–Bonferroni then multiplies by the family size and still gets zero.

**Produces.** `results/<run_id>/ensemble.json`, `oracle_bounds.json`, `statistical_validation.json`.

**Do not report new metrics** until OOF provenance, untouched-test predictions and the full
2000-iteration validation have all completed.

---

## 06_deployment — QAT, export, and the demo

*§4.6, Tables 3.3 / 4.14 / 4.15* · `pipelines/06_deployment.toml`

**Needs.** 05 complete: a calibrated selector and the fusion pool checkpoints, which together form the
frozen FP32 **teacher**.

**Steps.**

```powershell
conda run -n koa_project koa qat train --pool fusion --run-id run_001
conda run -n koa_project koa qat export --output artifacts/run_001/deploy/student.onnx
conda run -n koa_project koa records validate --all
conda run -n koa_project koa serve --check
```

`qat train` distils the teacher into a mixed-precision student under the six-term objective. The
precision profile is uneven on purpose (Table 3.3): the X-ray encoder is INT8 fake-quantised, the
selector is INT8 with an FP32 fallback, the MRI/RCKF branch stays FP16-mixed because it carries the
variance and gain estimates the reliability argument rests on, and the CORN head is **never** quantized
— four chained conditional sigmoids compound an early rounding error across every later class
posterior.

`qat export` writes the seven-field `qat-v2` contract to ONNX. Output names derive from the contract
dataclass, so they cannot drift out of order. Export **degrades gracefully**: with `onnx` absent it
reports `exported: false` with a reason and never raises.

`records validate` is status-aware in both directions: a `complete` record with an empty metric block
is an interrupted run mislabelled as finished, and a `pending` record carrying numbers is rejected —
that is precisely how transcribed or invented figures would acquire the authority of machine output.
It also scans every record string for placeholder words outside a two-key prose allowlist.

`serve --check` builds the API, exercises the mock predictor against the response contract the
workbench actually reads, and confirms `results/` is readable. It binds no socket — a start-up check
that needs a free port is a check of the port.

**Serving it for real:**

```powershell
conda run -n koa_project koa serve --host 127.0.0.1 --port 8000
cd frontend; npm run dev          # http://127.0.0.1:5173
```

| Route | Behaviour |
|---|---|
| `GET /health` | Service liveness plus `runtime: {configured, exists, modelId}` — two independent booleans, never one conflated one. |
| `GET /api/profiles` | The four deployment profiles. Every `fallback` resolves to a profile in the list. |
| `GET /api/results` | Run ids, listed from `results/` on disk minus `published`. No hardcoded whitelist. |
| `GET /api/results/<run_id>` | That run's curated record tree. 404 with the available ids for anything else. |
| `POST /api/predict` | JSON body → always the mock (a JSON body carries no pixels). Multipart → the torch runtime when a checkpoint is configured **and** exists, otherwise the mock with an explicit `runtime: "mock"`. |
| `POST /api/feedback/export` | Echo. **Persists nothing** — the workbench's queue lives in `localStorage` and is exported client-side. |

MRI multipart field names mirror the frontend exactly: a complete pair goes out as **`mri_t2` +
`mri_r2`**, a lone channel as **`mri`**. A lone channel is accepted and recorded but cannot form the
`(2, D, H, W)` tensor, so the case runs the X-ray route and reports `missingModalityFallback`.

Setting `KOA_V7_DEPLOYMENT_CHECKPOINT` switches multipart uploads to the real runtime. It is read once
at construction and the runtime is a process-wide singleton, so **changing it needs a restart**.
`KOA_V7_DEVICE` accepts `auto` (default), `cpu` or `cuda`.

---

## Where output goes

| Tree | Contents | Rule |
|---|---|---|
| `artifacts/<run_id>/` | Checkpoints, OOF predictions, logs, exports. | Auto-created. Heavy. Never curated by hand. |
| `results/<run_id>/` | Eight curated JSON records. | **Scaffold before training.** The trainer will not create it. |
| `results/published/` | Chapter 4's figures, transcribed from the PDF. | Read-only reference for tests. **Never copy a value out of it into a run record.** |
