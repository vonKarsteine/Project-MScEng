# Project-MScEng — multimodal Kellgren–Lawrence grading

[![tests](https://github.com/vonKarsteine/Project-MScEng/actions/workflows/tests.yml/badge.svg)](https://github.com/vonKarsteine/Project-MScEng/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](DEPENDENCIES.md)
[![PyTorch 2.5.1](https://img.shields.io/badge/pytorch-2.5.1%2Bcu118-ee4c2c.svg)](DEPENDENCIES.md)

Five-class Kellgren–Lawrence grading of knee osteoarthritis from **paired X-ray and MRI**
(quantitative T2 and R2 volumes), packaged as the engineering artefact behind Chapters 3 and 4 of an
HKU DASE7099 dissertation. Three contributions sit on top of a conventional ordinal-classification
baseline: **RCKF**, which recasts multimodal fusion as a Bayesian measurement update so the model
estimates *per case* how far the volumetric evidence can be trusted; **C-MODES**, which learns from
out-of-fold predictions alone when to route a case away from the default model and when
not to; and a **deployment QAT** stage that distils the FP32 composite into a mixed-precision student
under a six-term objective, so the quantized model inherits the teacher's cross-modal trust policy
rather than merely its answers.

## Repository map

| Path | What it holds |
|---|---|
| `src/koa_multimodal/` | All library code. Twelve packages in a strictly layered graph — see `AGENTS.md`. |
| `src/koa_multimodal/cli/` | The `koa` console script. There is no `scripts/` directory. |
| `src/koa_multimodal/api/` | The stdlib demo API behind the workbench. |
| `configs/default.toml` | Every tunable, one typed field each. `configs/pools/*.toml` bind candidate pools. |
| `pipelines/*.toml` | The six dissertation sections as declarative manifests, walked by `koa pipeline`. |
| `frontend/` | React + Vite + Konva clinical workbench. |
| `artifacts/` | Heavy machine output (checkpoints, OOF predictions, exports). Auto-created, never curated. |
| `data/` | The OAI subset. **Empty here** — see *Status*. |
| `tests/` | Contract tests. They are the deliverable's proof, since no training is run. |
| `docs/` | `methodology.md`, `runbook.md`, `dissertation_alignment.md`. |

## Install

One conda environment, `koa_project` (Python 3.9.25). The pinned scientific stack is already present;
three packages and the editable install are all that is needed:

```powershell
conda run -n koa_project pip install nibabel pytest onnxruntime
conda run -n koa_project pip install -e .
```

`environment.yml` rebuilds the whole environment from scratch elsewhere. Exact versions and the two
places they diverge from the dissertation are in `DEPENDENCIES.md`.

## Verification run book

This checkout ships **no data and no weights**, and no training is run in it. Everything below is
therefore the actual proof that the package is sound: contract tests, synthetic forward passes over
every route, resolved-but-unexecuted plans, and a live API. Run in order; every command should exit 0.

```powershell
conda run -n koa_project python -c "import koa_multimodal; print(koa_multimodal.__version__)"
conda run -n koa_project pytest -q
conda run -n koa_project koa check env
conda run -n koa_project koa check pipeline --synthetic
conda run -n koa_project koa check data --root data --allow-empty
conda run -n koa_project koa pipeline list
conda run -n koa_project koa pipeline run 03_fusion --dry-run
conda run -n koa_project koa records validate --all
conda run -n koa_project koa serve --check
cd frontend && npm ci && npm run build
```

What each step establishes:

| Command | What passing it proves |
|---|---|
| `import koa_multimodal` | The editable install resolves and the package imports with no `sys.path` help. |
| `pytest -q` | The layering graph is acyclic, the config is a bijection with `default.toml`, the `CandidateOutput` contract holds, and the code's constants agree with the dissertation's tables. |
| `koa check env` | Interpreter, torch, CUDA and every optional dependency, as actually installed. |
| `koa check pipeline --synthetic` | All five fusion routes **and** the C-MODES router forward on random tensors: shapes, posteriors summing to one, and the missing-MRI fallback firing. No data, no weights, no network. |
| `koa check data --root data --allow-empty` | The layout auditor runs and reports the dataset as absent rather than crashing — the expected state here. |
| `koa pipeline list` | The six section manifests parse. |
| `koa pipeline run 03_fusion --dry-run` | A stage plan fully resolves against the config — batch sizes, accumulation, objective weights — without executing anything. |
| `koa records validate --all` | Every record tree is internally consistent **and** `run_001` carries no metrics, which is what makes "nothing was trained here" machine-checked rather than asserted. |
| `koa serve --check` | The demo API builds, the mock predictor produces a response matching the shape the workbench reads, and `results/` is readable. Binds no socket. |
| `npm ci && npm run build` | The workbench compiles. |

To see the API for real:

```powershell
conda run -n koa_project koa serve --host 127.0.0.1 --port 8000
```

then `GET /health`, `GET /api/profiles`, and `POST /api/predict`. With no checkpoint configured every
prediction is served by the deterministic mock and says so in `metadata.runtime`. Setting
`KOA_V7_DEPLOYMENT_CHECKPOINT` to a real checkpoint switches multipart uploads to the torch runtime;
the variable is read once at start-up, so changing it needs a restart.


## Further reading

- `docs/methodology.md` — the three contributions with their equations, cross-referenced to modules.
- `docs/runbook.md` — what each pipeline stage needs, produces, and depends on.
- `docs/dissertation_alignment.md` — every place the code and the write-up disagree, and which one is right.
- `AGENTS.md` — contribution conventions and the layering rule.
- `DEPENDENCIES.md` — the verified environment.

## Data, licence and attribution

The Osteoarthritis Initiative (OAI) imaging data this work is trained on is **not** contained in or
distributed with this repository; access is governed by the OAI Data Use Agreement, and `data/` ships
empty by design.

The code is MIT-licensed (`LICENSE`). `ACKNOWLEDGMENTS.md` credits the data source and the methods
this work builds on — CORN, SimSiam, FedYogi, InfoNCE, Grad-CAM and the pretrained backbones — and
states which parts are contributed here: RCKF, C-MODES, the deployment QAT profile, and the
out-of-fold provenance scheme. `CITATION.cff` gives the citation metadata.
