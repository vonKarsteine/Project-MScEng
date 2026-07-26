# Dissertation alignment

Every place the code and the dissertation differ, stated rather than reconciled quietly.

A gap between a write-up and its implementation is not automatically a defect in either. Some of these
are places the text needs correcting; some are places the implementation is constrained by hardware
the text does not describe; some are places the text names two parameters where only one is
identifiable. What they have in common is that silently closing the gap in the *code* would make the
dissertation describe a run that did not happen, and silently closing it in the *prose* would hide a
real engineering constraint. So they are listed.

`tests/test_dissertation_alignment.py` pins the agreements that *do* hold, by reading the transcribed
tables under `results/published/` and comparing them against the package's constants. Nothing below is
a test failure; all of it is documented divergence.

---

## 1. pi uses Softplus, not the GELU the methodology names

**Text:** §3.2 names GELU as the activation inside the monotone variance estimator pi.
**Code:** `koa_multimodal/fusion/rckf.py::MonotoneVarianceEstimator` uses **Softplus**.
**The text needs the correction, not the code.**

pi exists to guarantee one property: measurement variance is coordinate-wise **monotone
non-decreasing** in the residual, so that a *larger* cross-modal disagreement can never be read as
*higher* modality credibility. That guarantee needs two ingredients simultaneously — positive weights,
supplied structurally by `PositiveLinear` (`softplus(B)`, eq. 3.12), and a **non-decreasing
activation**.

GELU is not non-decreasing. It dips below zero and reaches a local minimum near `x = -0.75` before
recovering. Composed into the stack, it makes the map non-monotone, and the guarantee pi is named for
is simply false. Softplus is monotone everywhere and satisfies the requirement exactly.

This is not a case of two acceptable choices. With GELU there is a residual band over which increasing
disagreement *decreases* the estimated variance, which increases the Kalman gain — the model would
trust the MRI measurement more precisely where the two modalities agree less. That inverts the
contribution.

`"gelu"` remains selectable via `[rckf] monotone_activation` for ablation only, and `RckfConfig.validate`
emits a `KoaWarning` when it is chosen (promoted to an error under pytest by `pyproject.toml`).

**Suggested correction to the text:** replace "GELU" with "Softplus" in the eq. 3.11 description, and
state monotonicity of the activation as a requirement rather than an incidental choice.

---

## 2. Global batch size 16 is reached by accumulation, not by a physical batch

**Text:** Table 4.3 states a global batch size of 16.
**Code:** per-modality physical batches of **4 / 1 / 4** (X-ray / MRI / fusion) with gradient
accumulation **4** on both volumetric paths.

The effective size matches — `4 x 4 = 16` for MRI and fusion — and `koa pipeline run ... --dry-run`
reports `batchSize`, `gradientAccumulation` and `effectiveBatchSize` side by side so the arithmetic is
visible rather than implied. `koa_multimodal/training/stages.py::StageSpec.gradient_accumulation`
carries the reason in its docstring.

The constraint is decode cost, not modelling. A `(2, 32, 384, 384)` float32 volume is decoded from two
NIfTI files and trilinearly resampled per sample per epoch on CPU (`[training] num_workers = 0` on
Windows, deliberately). At that cost a physical MRI batch above 1 buys nothing and spends memory that
the anisotropic stride-`(1,4,4)` stem of `MriEncoder` was designed to conserve. X-ray, which has no
such cost, runs at a real batch of 4 with no accumulation at all — so the "global batch size" in
Table 4.3 is not one number describing one thing.

**Consequence to be aware of:** gradient accumulation and a true larger batch are not identical under
batch-dependent normalisation. It is why `MriEncoder` uses GroupNorm and `SimSiamMriCandidate` uses
LayerNorm rather than the canonical SimSiam BatchNorm1d — at batch size 1 a batch statistic is a
zero-variance statistic. Those are not stylistic substitutions and should not be "fixed" back.

---

## 3. Table 4.4's RTX 4090 GPU-hours do not transfer

**Text:** §4.1.4 and Table 4.4 state an NVIDIA RTX 4090 with 24 GB, and report GPU-hours against it.
**Reality:** this checkout's development machine is a different, smaller device. `koa check env`
reports what it actually is.

Every wall-clock and GPU-hour figure in Table 4.4 is therefore **not reproducible here**, and no timing
claim anywhere in this repository is made against it. This is not merely a slower-machine caveat: the
memory budget shapes the architecture. The stride-`(1,4,4)` stem, `mri_batch_size = 1`, and the
accumulation scheme in §2 above are all consequences of a small budget. They are conservative on a
4090 and necessary on the actual card; the reported timings are the other way round.

`results/published/SOURCE.md` records the same, and states that the transcribed figures were produced
by an earlier checkout on the hardware §4.1.4 describes.

---

## 4. tau_s and c_switch are OOF-calibrated outputs, and only their sum is identifiable

**Text:** §3.3.4 names two parameters, a switch threshold `tau_s` and a switch cost `c_switch`.
**Code:** both are **absent from `configs/default.toml`** on purpose, and both are produced by
`ensemble/cmodes/calibration.py` and persisted into the selector checkpoint's `hyperparameters`.

Two separate points.

**They are outputs, not inputs.** They are chosen by a sweep over out-of-fold predictions — never
validation, never test, because calibrating on the evaluation set leaks it into the routing policy.
Putting them in the config file would present a calibration result as a configuration choice, and
would let someone "tune" the deployed routing behaviour by editing a TOML file with no OOF evidence
behind the new value. The config's docstring says so; `CmodesConfig` carries the note in place of the
fields.

**Only the sum enters the rule.** The decision is `best_alternative_score > tau_s + c_switch` — there
is no other appearance of either term. So the model is not identifiable in `(tau_s, c_switch)`; it is
identifiable in `threshold = tau_s + c_switch`. `CMODESRouter.threshold` exposes exactly that, and the
calibration sweep deduplicates candidate pairs on `round(tau + cost, 6)` rather than sweeping a
two-dimensional grid it cannot distinguish points in. The pair is still stored separately because the
methodology names them separately and the nominal deployment cost (0.04) is a meaningful default.

**Suggested clarification to the text:** state that the identifiable parameter is the sum, and that
the split into a threshold and a cost is interpretive.

---

## 5. Switch precision is ordinal error reduction, not a wrong-to-right flip

**Text:** §3.3 / Table 4.10 report "switch precision".
**Code:** `koa_multimodal/stats/metrics.py::switch_precision_components`, default `mode="ordinal"`:

```python
before = abs(default[i] - truth[i])
after  = abs(routed[i]  - truth[i])
if   after <  before: improved += 1
elif after >  before: degraded += 1
else:                 neutral  += 1
switch_precision = improved / (improved + degraded + neutral)
```

The criterion is `|routed - y| < |default - y|`, not `routed == y and default != y`. This matches the
ordinal risk the selector is actually trained on: the C-MODES target charges
`lambda * |y_hat - y| / 4` alongside `-log p(true)`, so a switch from KL4 to KL3 on a true KL2 removes
real risk and *should* count as an improvement, even though both predictions are wrong.

**Neutral switches count against it.** They sit in the denominator, not outside it. A route change that
lands on the same grade cost a switch and bought nothing, and a metric that ignored those would reward
churn.

**Always report `switchCount` beside it.** Two reasons. First, the value is **NaN when nothing
switched** — not 0, not 1 — because precision over an empty set is undefined; the static-ensemble rows
of Table 4.10 report 0.000 because they never switch, which is a different statement from a policy
that switched and was right none of the time. Second, switch precision is trivially maximised by
almost never switching, so it is used in calibration as a *feasibility constraint* with QWK as the
objective, never as the objective itself.

A `mode="exact"` correct-vs-incorrect variant exists for comparison but is not what is reported.

---

## 6. The 3-D MRI backbones stay randomly initialised

**Text:** `§4.1.4` states explicitly that the volumetric backbones are randomly initialised.
**Code:** `[model] mri_pretrained = false`, and this is the default everywhere.

Kinetics-400 initialisation *is* implemented — `build_base_candidate(..., mri_pretrained=True)` with
the rescaled stem adaptation in `koa_multimodal/models/inflate.py` — and turning it on makes that
sentence in the text false. It is available for ablation and is not the reported configuration.

The asymmetry with the X-ray branch is deliberate and runs the other way. `TimmXrayBackbone` **raises**
when pretrained weights are requested and cannot be had, because §4.1.4 records all five X-ray encoders
as ImageNet-initialised and silently falling back to random would leave Table 4.5 describing a run that
did not happen. The MRI backbones **warn and fall back**, because random initialisation *is* the
reported MRI configuration — the fallback lands on the documented default rather than away from it.

Related: `mri_simsiam_ssl` is the locally defined `MriEncoder`, tagged `mri-encoder-v2`. The tag is
compared on checkpoint grafting and a stale one is refused, because branch loads that ran with
`strict=False` could not distinguish a wholesale key mismatch from success.

---

## 7. The subject-clustered bootstrap assigns each subject its *highest* KL grade

**Text:** §3.5 describes a stratified, subject-clustered bootstrap.
**Code:** `koa_multimodal/stats/resampling.py`, constant
`GROUP_STRATUM_RULE = "subject_assigned_to_highest_kl_grade"`, published in every payload as
`groupStratumRule`.

Stratification and clustering pull against each other here, and there is no exact resolution. A
patient contributes two knees, and those two knees routinely carry **different** KL grades — so a
subject has no single label to be stratified by. Clustering is non-negotiable (treating two knees of
one patient as independent understates the interval width), and stratification is non-negotiable
(KL4 is 22 of 673 test pairs; a uniform resample would draw pseudo-cohorts containing none of it).
Something has to give, and what gives is label purity within a stratum.

Assigning the **maximum** grade is the conservative choice for this application: it keeps the severe
grades populated in the strata that need them, and it errs toward over-representing advanced disease
rather than diluting it. The alternatives are worse — the minimum systematically empties the KL4
stratum, and the mean is not a grade.

The rule is recorded in every bootstrap and paired-comparison payload rather than left implicit,
because an interval computed under a different clustering convention is not comparable to one computed
under this.

**One inconsistency worth knowing:** the same idea appears under two different spellings. The fold
manifest (`data/folds.py`) publishes `groupStratumRule = "max_label_within_group"` while the bootstrap
publishes `"subject_assigned_to_highest_kl_grade"`. They describe the same max-grade rule for two
different purposes — fold construction and resampling strata — but a reader diffing payloads will see
two strings.
