# Methodology

The three contributions of Chapter 3, with their equations and the modules that implement them. This
is a reading guide, not a restatement of the dissertation: where the code and the text disagree,
`dissertation_alignment.md` records which one is right and why.

Everything below sits on one shared foundation. KL grading is **ordinal** — mistaking KL2 for KL4 is
clinically far worse than mistaking it for KL3 — so every model that owns a decision layer terminates
in a CORN head (`koa_multimodal.core.ordinal.CornOrdinalHead`) emitting `K - 1 = 4` conditional
threshold logits `P(y > k | y > k-1)`, converted to five class posteriors by the chain rule. Model
selection is on validation **QWK**, not accuracy, for the same reason.

Every model returns one type, `CandidateOutput` (`koa_multimodal.core.contract`), carrying
probabilities, optional threshold logits, prediction, uncertainty, metadata and a typed diagnostic
`Trace`. The two posterior-averaging routes — the OOF stacker and the C-MODES router — own no ordinal
head and leave `threshold_logits` as `None`; `require_threshold_logits()` is the guard that makes
applying a CORN objective to a class-probability vector impossible rather than merely wrong.

---

## 1. RCKF — Residual-Calibrated Kalman Fusion

`koa_multimodal/fusion/rckf.py` · §3.2 · Table 4.9

### The problem it answers

The three traditional baselines (`fusion/traditional.py`) fuse by concatenation, gating and cross
attention, and the empirical ordering is the argument for doing something else: cross attention, the
most expressive of the three, is *worse* than gating (0.788 vs 0.795 accuracy, Table 4.7). A more
sensitive interaction module amplifies heterogeneous cross-modal noise as readily as signal.
Contrastive alignment (`fusion/contrastive.py`, Table 4.8) improves on that by implicitly filtering
the uncorrelated component — but it applies the *same* pull to every sample. Neither has any notion of
how far *this particular* volume can be trusted. That is the gap.

### The formulation

Fusion is recast as a Bayesian measurement update on a latent knee state `z`. The X-ray feature is the
**prior**, the MRI feature projected into the same latent space is the **measurement**:

```
f_x = z + eps_x,    eps_x ~ N(0, P_x),   P_x = sigma_x^2 I     (eq. 3.4)
u_m = W_m f_m = z + eps_m,   eps_m ~ N(0, R_m)                 (eq. 3.5)
```

### Why the prior variance is *fixed* at sigma_x² = 0.15

`[rckf] prior_variance = 0.15`, passed to `RCKFBlock.__init__` and materialised with `torch.full_like`.
It is a constant, not a parameter, and that is the load-bearing design decision of the whole module.

If both branches could learn their own variance they would compete for the Kalman gain, and the gain
would stop meaning *"how much do I trust the measurement"* — it would mean only *"what ratio minimises
the training loss"*, which any pair of freely-scaled variances can produce at any gain. Pinning one
side fixes the units. The KL label is native to radiographs, so the radiograph is the branch whose
reliability needs no estimating; the MRI branch is the one whose noise is heterogeneous, because
acquisition is asynchronous and quantitative maps vary in quality. Fixing `P_x` is what turns `K` from
a learned mixing weight into a *calibrated statement about the measurement*.

### Two residuals, doing two different jobs

This is the easiest thing in the module to get wrong, because conflating them passes every shape check.

```
e  = | norm(f_x) - norm(u_m) |          scale-free   -> pi only      (eq. 3.8)
nu = f_x - u_m                          raw          -> update + NLL (eq. 3.10)
```

`e` is L2-normalised on both sides *before* the difference. Two independently trained encoders have no
reason to share a feature-norm scale, and without normalisation `pi` would respond to that discrepancy
rather than to disagreement. So the uncertainty estimator sees **direction**, not magnitude.

`nu` is the raw innovation. It drives the posterior update and the Gaussian innovation likelihood,
where the actual magnitude is exactly what matters. Feeding `nu` to `pi`, or updating with `e`, is
finite, plausible and quietly destroys the calibration.

### R̂ = pi(e), monotone by construction, bounded to [0.05, 2.0]

`MonotoneVarianceEstimator` is a stack of `PositiveLinear` layers — the weight is stored free and used
as `softplus(B)` (eq. 3.12), so positivity is structural rather than a penalty training could trade
away — interleaved with a non-decreasing activation, then squashed:

```
R_hat = R_floor + (R_ceil - R_floor) * sigmoid(net(e))          (eqs. 3.11, 3.13)
```

Monotonicity is the property the module exists to provide: a *larger* cross-modal disagreement must
never be read as *higher* modality credibility. It needs positive weights **and** a non-decreasing
activation, together. The floor `R_floor = 0.05` encodes irreducible baseline measurement noise — zero
variance would claim a perfect measurement. The ceiling `R_ceil = 2.0` stops the variance diverging,
which would drive the gain to zero and silently sever the MRI stream altogether.

> The methodology text names **GELU** for the activation. The code uses **Softplus**, and the code is
> right: GELU dips below zero around `x = -0.75`, so it is non-monotone and voids the guarantee. See
> `dissertation_alignment.md` §1. `"gelu"` remains selectable via `[rckf] monotone_activation` for
> ablation and emits a `KoaWarning`.

### The update

```
S = P_x + R_hat                                                  innovation covariance
K = P_x / S                                                      (eq. 3.14)
z_post = LayerNorm( f_x + K * (u_m - f_x) )                       (eq. 3.15)
```

`K` is bounded in `(0, 1)` by construction and reads directly as *the fraction of the answer that came
from MRI*. It is one of the seven fields of the deployment contract for exactly that reason.

The training objective adds the Gaussian innovation NLL (`gaussian_innovation_nll`, second term of
eq. 3.16) at weight `alpha_res = 0.1`:

```
L = L_CORN + alpha_res * 0.5 * [ nu^T S^-1 nu + log|S| ]
```

The `log|S|` term is what makes this a *likelihood* rather than a penalty on the residual. Driving the
innovation to zero reduces it; so does honestly widening the variance when the modalities disagree.
Without that second term the estimator would simply learn to report small variances.

### Missing MRI: a principled degenerate update

`RCKFBlock.fallback`. MRI is routinely unavailable in the clinical workflow this targets, so every
fusion route accepts `mri=None`. With no measurement the best available observation *is* the prior, so
the innovation is exactly zero and `z_post = f_x` for any gain. The variance is pinned at the
**ceiling** and the gain reported at the value that ceiling implies, `P_x / (P_x + R_ceil)`.

Reporting a consistent pair rather than zeroed features matters downstream: the QAT reliability term
and the deployment contract both read variance and gain, and a fabricated zero would tell them the
model was confident in a measurement it never received. The config cross-check
`variance_ceiling > prior_variance` exists to stop the degenerate gain exceeding a live one — a missing
modality must never look *more* trustworthy than a present one.

---

## 2. C-MODES — Calibrated Multimodal OOF Dynamic Ensemble Selector

`koa_multimodal/ensemble/cmodes/` · §3.3 · Tables 4.10–4.11

### The problem it answers

The oracle bound (`ensemble/stacking.py::oracle_upper_bound`, Table 4.11) reaches 0.897 accuracy on
the X-ray pool and 0.938 on the bimodal pool — far above any single member. What separates the
deployed ensemble from perfect is therefore a **selection** problem, not a capacity one. That is the
argument for routing rather than for bigger models, and it is why the oracle is reported as a
*measure of headroom*, never as a result.

### The 16-D pairwise feature

`ensemble/cmodes/features.py::RiskFeatureBuilder`. One vector per **(sample, route) pair**, of fixed
width `2 * (K + 3) = 16` whatever the pool size, because it describes a pair and not the pool:

| Slots | Block | Contents |
|---|---|---|
| 0–4 | context `a_i` | the default route's five posteriors |
| 5 | context | default expected grade `sum_c c * p_c` |
| 6 | context | default entropy |
| 7 | context | default top-2 margin |
| 8–12 | differential `b_ik` | per-class `abs(p_k - p_default)` |
| 13 | differential | expected-grade step `E_k - E_default` (signed) |
| 14 | differential | top-2 margin delta (signed) |
| 15 | differential | the default margin's boundary bin, `bucketize(margin, [0.10, 0.30])` |

One shared scalar MLP (`CMODESSelectorModel`, 16 → 64 → 1) scores **every** route with the **same**
weights. Sharing them is what makes the selector permutation-equivariant over the pool: reordering the
candidates reorders the scores and changes nothing else. A per-route head would instead let the
selector memorise "route 3 is usually good", which is precisely the static-ensemble failure mode
C-MODES exists to improve on.

### The regression target: a risk *differential*

`ensemble/cmodes/risk.py`. Risk is ordinal-aware, not accuracy-based:

```
L(M, y) = -log p_M(y) + lambda * |y_hat - y| / (K - 1)      lambda = 1.0, K - 1 = 4
Delta_ik = L(M_0(x_i), y_i) - L(M_k(x_i), y_i)              default minus candidate   (eq. 3.19)
```

A positive differential means switching to route `k` removes risk. The selector regresses `Delta`
under **smooth L1**, not MSE: a confidently-wrong `-log p` has a heavy tail, and squared error would
let a handful of samples dictate the whole routing policy.

The `-log p(true)` term rewards calibrated confidence and the `|y_hat - y| / 4` term charges for
ordinal distance, so a route that is wrong by one grade is preferred over one that is wrong by three.

### Training: a simulated federated loop, FedYogi

`ensemble/cmodes/federated.py`. Each candidate index is treated as a pseudo-client holding its own
column of the feature tensor; per round every client takes a few local SGD steps from the shared
global model, and the server aggregates.

Two sign conventions are inverted relative to the usual formulation, together, and both must be kept:

```
Delta_k = theta_global - theta_local           GLOBAL minus local          (eq. 3.23)
g_t     = sum_k w_k Delta_k                    aggregated pseudo-gradient  (eq. 3.24)

m_t = beta_1 m_{t-1} + (1 - beta_1) g_t
v_t = v_{t-1} - (1 - beta_2) sign(v_{t-1} - g_t^2) g_t^2
theta_{t+1} = theta_t - alpha * m_t / (sqrt(v_t) + tau)      SUBTRACT the step  (eq. 3.25)
```

with `beta_1 = 0.9`, `beta_2 = 0.999`, `tau = 1e-3`, `alpha = 0.5`. Because the delta is already
negated relative to convention, the server subtracting the step is a descent. **Flipping either sign
alone passes every shape check and silently ascends the loss**, which is why `tests/test_cmodes.py`
asserts the convention directly rather than trusting a comment.

Yogi's controlled second moment — `v` can only decrease by a bounded amount per step — is what stops
one large client spiking the denominator and stalling every later update.

### Inference: the switch rule

`ensemble/cmodes/routing.py::switch_decision`. Two mechanisms guard against volatile routing:

```
best = max_{k != default} score_k          the default column is masked to -inf
switch  iff  best > tau_s + c_switch                                (eqs. 3.26-3.27)
```

**The default is masked out of the argmax.** "Stay" is the fallback, not a competitor — it must never
win by a rounding error, and it must never *lose* to one either.

**A switch must clear a barrier.** `tau_s` and `c_switch` are named separately in the methodology but
enter the rule only through their sum, so the identifiable free parameter is
`threshold = tau_s + c_switch`. Both are **outputs of out-of-fold calibration**
(`ensemble/cmodes/calibration.py`), persisted in the selector checkpoint's `hyperparameters` — they are
deliberately absent from `configs/default.toml`, because they are not inputs to the process that
produces them. Calibrating on validation or test predictions would leak the evaluation set into the
routing policy.

Selecting a posterior destroys the conditional threshold chain, so the router builds its output with
`CandidateOutput.from_posterior`, never `from_corn`.

### The OOF provenance gate

The methodological spine, and the thing most easily weakened by accident. Anything that *trains* a
selector or stacker loads artifacts with `require_oof=True`, which rejects ordinary validation
predictions outright (`ensemble/artifacts.py`). Each artifact carries per-row `subjectIds` and
`foldIds` plus a per-fold `trainSubjectHash` — sha256 over the sorted, de-duplicated training subject
ids — so a reader can verify *cryptographically* that fold-*k* predictions came from a model trained on
exactly the non-*k* subjects. Folds are subject-grouped and grade-stratified
(`data/folds.py::grouped_stratified_fold_ids`), because a patient contributes two knees and splitting
them across folds leaks.

Producing those artifacts requires genuine per-fold retraining, not relabelling one validation run.
**Never relax `require_oof`** — produce V2 artifacts instead.

---

## 3. Deployment QAT

`koa_multimodal/deploy/` · §4.6, Tables 3.3 / 4.14 / 4.15

### The precision profile is deliberately uneven

Table 3.3. Not everything is quantized, and the exceptions are the point:

| Component | Precision | Why |
|---|---|---|
| X-ray encoder | INT8 fake-quant | The computational bulk, and robust to it. |
| C-MODES selector | INT8 with FP32 fallback | Small, but its output volatility is what the fallback contains. |
| MRI / RCKF branch | FP16 mixed | Carries the variance and gain estimates the whole reliability argument rests on. |
| CORN ordinal head | **FP32 always** | Four chained conditional sigmoids compound an early rounding error across every later class posterior. |

`FusionBase.forward` enforces the last row structurally: `self.head(latent.float())`. The measurement
branch runs under a single `measurement_autocast` context spanning both the encoder and the Kalman
update, and returns to FP32 before the head.

Fake quantisation is straight-through (`deploy/quantize.py`): `x + (q(x) - x).detach()`, so the
forward value is quantised and the gradient is the identity. The scale comes from an EMA min/max
observer. **Observation is decoupled from `Module.training`** — a separate `observing` flag — because
the textbook `self.training` gate is correct only if every quantized component actually runs in
training mode during QAT — and the router does not: it scores the selector under `eval()`. Gating on
`self.training` there would freeze that observer's range at whatever the first minibatch contained.
Nothing would fail, every loss would stay finite, and the deployed INT8 selector would silently carry
a range calibrated on a single batch.

### The six-term objective

Distillation transfers the whole deployment contract, not just the answer:

```
L = L_CORN
  + lambda_1 * L_KD          (temperature tau = 3.0)     class posteriors
  + lambda_2 * L_boundary    (KL1-weighted)              the minority boundary class
  + lambda_3 * L_ordinal                                 ordinal distance
  + lambda_4 * L_reliability                             R_hat and K alignment
  + lambda_5 * L_selector                                route CE + score MSE
```

with `lambda = (0.5, 0.2, 0.1, 0.1, 0.1)`. The reliability and selector terms are what make this more
than ordinary knowledge distillation: the student is supervised on the teacher's *measurement variance
and Kalman gain*, and on its *routing scores*, so it inherits the cross-modal trust policy rather than
imitating its outputs. A student that agreed on every grade while disagreeing on every gain would be a
different deployed system wearing the same accuracy.

The teacher is frozen: `train()` permanently re-forces `teacher.eval()`, and optimizers are built on
the student's parameters only.

### The seven-field deployment contract

`deploy/contract.py`, schema `qat-v2`. A deployed grader does not return a grade — it returns the
grade *plus the evidence a clinician needs to decide whether to believe it*:

```
probabilities, posterior_latent, measurement_variance, kalman_gain,
selector_scores, route, missing_modality_fallback
```

**The order is structural.** `as_tuple()`, `CONTRACT_FIELD_NAMES` and `ONNX_OUTPUT_NAMES` are all
derived from `dataclasses.fields(DeploymentContract)`, so a field cannot be inserted in one place and
missed in the others. Writing the same ordering out three times across two modules, held together by a
comment, would let an inserted field produce an ONNX graph whose output names are silently off by
one — which no shape check can catch, because every name still maps to a tensor of a plausible shape.

Absent branches are zero-filled at the correct batch width rather than omitted — an X-ray-only
candidate has no Kalman update, a single model has no selector scores — so the payload has fixed arity
whatever is behind it. That is what lets one ONNX signature and one API response shape serve every
configuration. Every default is built with `zeros_like` on a slice of `probabilities` rather than from
a Python `int` batch size, because the tracer would fold the latter into the graph as a constant.

### Serving it

`koa_multimodal/api/` — standard library only, two predictors (a labelled deterministic mock and the
real torch runtime), and Grad-CAM over the X-ray encoder's last convolutional stage as an **FP32
shadow path** (Table 4.14): a second, gradient-enabled, unconditionally-FP32 pass whose only product
is the saliency map, so explainability cannot perturb the grade and INT8 rounding cannot corrupt the
attribution. See `docs/runbook.md` §06 and the module docstrings.
