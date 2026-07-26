# Dependencies

One conda environment, `koa_project`. There is no second environment: training, tests, the CLI and
the demo API all run in it.

## Verified environment

Reported by `conda run -n koa_project koa check env` on the development machine. These are installed
versions, read from the modules themselves, not a wish list.

| Component | Version | Notes |
|---|---|---|
| Python | 3.9.25 | Sets the language floor: no `match`, no `StrEnum`, no `dataclass(slots=True)`, `from __future__ import annotations` in every module, and `Optional[X]` / `Dict` / `List` / `Tuple` rather than PEP 604/585 syntax. |
| torch | 2.5.1+cu118 | CUDA build 11.8, `torch.cuda.is_available()` → `True`. |
| torchvision | 0.20.1 | Supplies the `r3d_18` and `swin3d_t` video trunks. |
| timm | 1.0.24 | All five X-ray backbones. See the drift note below. |
| numpy | 2.0.2 | |
| opencv-python | 4.12.0.88 | CLAHE only. |
| Pillow | 11.1.0 | Image decode and the bicubic resize the recorded preprocessing used. |
| onnx | 1.19.1 | Optional `[deployment]` extra. |
| onnxruntime | 1.19.2 | Optional `[deployment]` extra. |
| nibabel | 5.3.3 | Optional. See below. |
| pytest | 8.4.2 | |
| tomli | 2.2.1 | The TOML parser below Python 3.11. |

`nibabel`, `pytest` and `onnxruntime` are the three the install step adds:

```powershell
conda run -n koa_project pip install nibabel pytest onnxruntime
conda run -n koa_project pip install -e .
```

## Optional dependencies, and what "optional" means here

**`nibabel` is genuinely optional.** `koa_multimodal.data.mri` falls back to a self-contained NIfTI-1
reader that honours the header's endianness, datatype and `scl_slope`/`scl_inter` scaling, so it
returns the array `nibabel` would have returned. That is why the fallback is silent rather than
warned: there is no numerical difference for a caller to be told about. It is still worth installing —
the fallback reads single-file `.nii`/`.nii.gz` only, and `nibabel` is the reference implementation.

**`opencv-python` is not optional in practice.** Without it CLAHE cannot run. It is installed here,
so this does not bite, but the failure mode is worth knowing: `[preprocessing] xray_require_clahe`
exists so a missing `cv2` can be made a start-up error rather than a silently skipped contrast
step, which would change the input distribution without changing any shape.

**`onnx` / `onnxruntime` gate only the export path**, which degrades to `exported: false` with a
reason and never raises. Both are installed here; nothing has been exported, because there is no
student checkpoint to export.

## Two documented drifts from §4.1.4

Recorded rather than quietly reconciled. `docs/dissertation_alignment.md` carries the full list;
these two are the dependency-shaped ones.

### 1. timm 1.0.19 reported, 1.0.24 installed

§4.1.4 states timm 1.0.19. The installed release is 1.0.24. The two are API-compatible for every
backbone this package builds — `timm.create_model(..., in_chans=1, num_classes=0, global_pool="avg")`,
`timm.get_pretrained_cfg(...)`, and the five pretrained tags
(`convnextv2_tiny.fcmae_ft_in22k_in1k_384`, `maxvit_tiny_tf_384.in1k`,
`swin_tiny_patch4_window7_224.ms_in22k_ft_in1k`, `tf_efficientnetv2_s.in21k_ft_in1k`,
`deit3_small_patch16_384.fb_in22k_ft_in1k`) resolve identically. No behavioural difference has been
observed, and nothing in the package works around a version difference.

The reason the exact release still matters is that timm's *weight index* changes between releases
independently of its API. `koa_multimodal.models.xray.timm_pretrained_available` checks the index
before constructing, so requesting ImageNet weights timm cannot supply is an error naming the
candidate rather than a random initialisation reported as pretrained — §4.1.4 records all five X-ray
encoders as ImageNet-initialised, and silently losing that would leave Table 4.5 describing a run
that did not happen.

### 2. GPU: RTX 4090 / 24 GB reported, a different device here

§4.1.4 and Table 4.4 state an NVIDIA RTX 4090 with 24 GB. The development machine for this checkout
is not that device — `koa check env` reports what it actually is.

**Table 4.4's GPU-hour estimates therefore do not transfer**, and neither does any memory headroom
argument derived from 24 GB. This is not a cosmetic difference: `[training] mri_batch_size = 1` with
gradient accumulation 4 is a consequence of volumetric memory pressure, and the anisotropic
stride-`(1,4,4)` stem of `MriEncoder` exists to hold activations inside a small budget. Those choices
are correct on the smaller card and would merely be conservative on a 4090; the reported wall-clock
numbers are the other way round and cannot be reproduced here at all. No timing figure in this
repository is claimed against Table 4.4.

## What is *not* a dependency

The demo API is standard library only: `http.server`, `email.parser`, `json`, `zlib`, `struct`,
`hashlib`, `base64`. No FastAPI, Flask, uvicorn or pydantic, and no image library on the mock path —
the PNG encoder in `koa_multimodal/api/predictors/heatmap.py` is twenty lines of `zlib` + `struct`.
§4.6.3 presents the deployment as a serverless-style prototype whose claim is that it needs
an interpreter and a checkpoint and nothing else; a web framework would falsify that claim while
changing nothing observable. See the module docstring of `koa_multimodal/api/server.py`.

The frontend's dependencies are its own (`frontend/package.json`): React 18, Vite 6, Konva /
react-konva, and `onnxruntime-web` reserved for a browser inference mode that is present in the
dropdown and reports that it is not implemented rather than faking a result.
