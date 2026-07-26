"""The real inference path: a trained checkpoint, loaded once, served in-process.

**Configured and present are two different questions.** Asking only whether
``KOA_V7_DEPLOYMENT_CHECKPOINT`` is a non-empty string lets a typo in the path
produce a manifest reading ``configured: true`` and a 503 whose message blames a
missing configuration that is, in fact, present. So :func:`runtime_manifest`
reports ``configured`` and ``exists`` separately, and
:meth:`TorchRuntime.from_environment` refuses to build anything unless the file is
really on disk -- "you set the variable" and "the file it names is there" cannot
be conflated.

**Preprocessing is the trainer's, not a copy of it.** The radiograph goes through
:func:`koa_multimodal.data.xray.load_xray_image` followed by
:func:`~koa_multimodal.data.xray.normalize_xray_imagenet_gray`, in that order and
with the same :class:`~koa_multimodal.config.schema.PreprocessingConfig` the
training run used; the volumes go through
:func:`koa_multimodal.data.mri.load_mri_pair`. Those loaders take paths, so the
uploaded bytes are written to a temporary file first. Reimplementing the pipeline
against ``BytesIO`` would be shorter and would silently drift -- the ordering
constraint that normalisation must follow augmentation, the uint8 round trip
before the bicubic resize, the per-channel clip windows -- and a served model
whose inputs differ from its trained inputs is wrong in a way no shape check sees.

**Checkpoint layout.** ``koa qat export`` is a planning stub in this checkout and
no weights ship with it, so the payload this module reads is stated here rather
than discovered:

.. code-block:: python

    {
      "schemaVersion": "koa-deployment-v1",
      "modelType": "single" | "composite",
      "featureDim": 64, "numClasses": 5,
      "quantization": "fp32",
      "rckf": {"priorVariance": .., "varianceFloor": .., "varianceCeiling": ..,
               "varianceGroups": .., "activation": "softplus"},
      # modelType == "single"
      "candidate": {"candidateId": .., "route": "rckf",
                    "xrayCandidateId": .., "mriCandidateId": ..},
      "modelStateDict": {...},
      # modelType == "composite"
      "candidates": [{"candidateId": .., "route": .., "modelStateDict": {...}}, ...],
      "defaultCandidateId": .., "tauS": .., "switchCost": ..,
      "selector": {"inputDim": 16, "hiddenDim": 64, "stateDict": {...}},
    }

Anything absent falls back to the configured default, and a state dict that does
not fit the model it is loaded into is rejected rather than partially applied --
:func:`koa_multimodal.core.checkpoint.load_module_state` exists for exactly that.

**The runtime is a module-level singleton.** The checkpoint is read once and the
model stays resident, so the environment is sampled exactly once per process:
changing ``KOA_V7_DEPLOYMENT_CHECKPOINT`` after the first prediction has no
effect until the server is restarted. :func:`reset_runtime` drops the cache for
tests; it is not an endpoint.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from koa_multimodal.api.predictors import heatmap as heatmap_lib
from koa_multimodal.config.loader import load_config
from koa_multimodal.config.schema import Config

#: The one environment variable that turns the real path on.
CHECKPOINT_ENV_VAR = "KOA_V7_DEPLOYMENT_CHECKPOINT"

#: ``auto`` (default), ``cpu`` or ``cuda``.
DEVICE_ENV_VAR = "KOA_V7_DEVICE"

MODEL_ID = "koa_multimodal_v7"
RUNTIME_NAME = "torch-v7"
GRAD_CAM_METHOD = "grad-cam"


# ----------------------------------------------------------------- environment


def _configured_path() -> Optional[Path]:
    raw = os.environ.get(CHECKPOINT_ENV_VAR, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def runtime_manifest() -> Dict[str, Any]:
    """What the server can say about the real path *without* loading anything.

    ``configured`` and ``exists`` are independent, and both are reported. A
    caller deciding whether to route a request to this runtime must require
    both: ``configured and exists``.
    """

    path = _configured_path()
    resolved = None
    exists = False
    if path is not None:
        try:
            resolved = path.resolve()
            exists = resolved.is_file()
        except OSError:
            # An unresolvable path (a bad drive letter, a broken link) is a
            # missing checkpoint, not a crash at /health.
            resolved = path
            exists = False
    return {
        "modelId": MODEL_ID,
        "runtime": RUNTIME_NAME,
        "envVar": CHECKPOINT_ENV_VAR,
        "configured": path is not None,
        "exists": exists,
        "checkpoint": str(resolved) if resolved is not None else None,
        "loaded": _runtime is not None,
        "note": (
            "configured reports only that the environment variable is set; exists "
            "reports whether the file it names is on disk. The runtime is built "
            "only when both are true, and it is cached for the life of the "
            "process -- changing the variable needs a server restart."
        ),
    }


def runtime_available() -> bool:
    manifest = runtime_manifest()
    return bool(manifest["configured"] and manifest["exists"])


def _resolve_device() -> "Any":
    import torch

    requested = os.environ.get(DEVICE_ENV_VAR, "auto").strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{DEVICE_ENV_VAR}=cuda was requested but torch reports no CUDA device"
            )
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- construction


def _rckf_kwargs(payload: Dict[str, Any], config: Config) -> Dict[str, Any]:
    """RCKF hyperparameters from the checkpoint, falling back to the config.

    They have to travel with the weights: the variance ceiling in particular is
    baked into the missing-MRI fallback gain, so serving a checkpoint under a
    different ceiling would report a Kalman gain the model was never trained to
    produce.
    """

    stored = payload.get("rckf") or {}
    return {
        "prior_variance": float(stored.get("priorVariance", config.rckf.prior_variance)),
        "variance_floor": float(stored.get("varianceFloor", config.rckf.variance_floor)),
        "variance_ceiling": float(
            stored.get("varianceCeiling", config.rckf.variance_ceiling)
        ),
        "variance_groups": int(stored.get("varianceGroups", config.rckf.variance_groups)),
        "activation": str(stored.get("activation", config.rckf.monotone_activation)),
    }


def _build_route(
    spec: Dict[str, Any], payload: Dict[str, Any], config: Config
) -> "Any":
    """One fusion route, built to match what the checkpoint says it is.

    When the entry names its two branch candidates the route is assembled through
    :func:`koa_multimodal.fusion.assembly.build_fusion` with ``pretrained=False``:
    the encoders are about to be overwritten by the checkpoint, so downloading
    ImageNet weights first would be a slow no-op that also requires a network.
    Otherwise a bare :class:`~koa_multimodal.fusion.rckf.RCKFFusion` over the
    reference encoders is built, which is what a smoke-exported checkpoint holds.
    """

    from koa_multimodal.fusion.assembly import build_fusion
    from koa_multimodal.fusion.rckf import RCKFFusion

    route = str(spec.get("route", "rckf"))
    candidate_id = str(spec.get("candidateId", route))
    xray_id = spec.get("xrayCandidateId")
    mri_id = spec.get("mriCandidateId")

    if xray_id and mri_id:
        model = build_fusion(
            route,
            str(xray_id),
            str(mri_id),
            model_cfg=config.model,
            rckf_cfg=config.rckf,
            pretrained=False,
        )
        model.candidate_id = candidate_id
        return model

    return RCKFFusion(
        feature_dim=int(payload.get("featureDim", config.model.feature_dim)),
        num_classes=int(payload.get("numClasses", config.model.num_classes)),
        candidate_id=candidate_id,
        **_rckf_kwargs(payload, config)
    )


def _build_router(payload: Dict[str, Any], candidate_ids: Sequence[str], config: Config):
    """The C-MODES router over the pool, when the checkpoint carries a selector.

    A composite checkpoint with no selector block is legitimate -- a pool served
    without a fitted selector -- and gets a router whose ``selector_model`` is
    ``None``. That router falls back to an uncertainty differential, which is a
    documented stand-in and is reported as such in the response diagnostics, not
    passed off as a trained routing policy.
    """

    from koa_multimodal.ensemble.cmodes.routing import CMODESRouter
    from koa_multimodal.ensemble.cmodes.selector import CMODESSelectorModel
    from koa_multimodal.core.checkpoint import load_module_state

    selector_spec = payload.get("selector") or {}
    selector = None
    if selector_spec.get("stateDict") is not None:
        selector = CMODESSelectorModel(
            input_dim=int(selector_spec.get("inputDim", 2 * (config.model.num_classes + 3))),
            hidden_dim=int(selector_spec.get("hiddenDim", config.cmodes.selector_hidden_dim)),
        )
        load_module_state(selector, selector_spec["stateDict"], strict=True)

    return CMODESRouter(
        selector_model=selector,
        candidate_ids=list(candidate_ids),
        default_candidate_id=str(payload.get("defaultCandidateId", candidate_ids[0])),
        tau_s=float(payload.get("tauS", 0.0)),
        c_switch=float(payload.get("switchCost", 0.04)),
    )


class TorchRuntime:
    """A loaded deployment checkpoint, ready to answer requests."""

    def __init__(
        self,
        *,
        models: List["Any"],
        router: Optional["Any"],
        device: "Any",
        checkpoint: Path,
        config: Config,
        quantization: str,
        selector_fitted: bool,
    ) -> None:
        self.models = models
        self.router = router
        self.device = device
        self.checkpoint = checkpoint
        self.config = config
        self.quantization = quantization
        self.selector_fitted = selector_fitted
        for model in self.models:
            model.to(device).eval()
        if self.router is not None:
            self.router.to(device).eval()

    # -- construction ------------------------------------------------------

    @classmethod
    def from_environment(cls, config: Optional[Config] = None) -> "TorchRuntime":
        from koa_multimodal.core.checkpoint import load_checkpoint, load_module_state

        path = _configured_path()
        if path is None:
            raise FileNotFoundError(
                f"{CHECKPOINT_ENV_VAR} is not set, so there is no deployment "
                "checkpoint to serve."
            )
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"{CHECKPOINT_ENV_VAR} points at {resolved}, which does not exist. "
                "The variable being set is not the same as the file being present; "
                "refusing to build a runtime around a path that resolves to nothing."
            )

        config = config or load_config()
        device = _resolve_device()
        # Trusted: this package's own checkpoints embed feature specs and
        # hyperparameters, which weights_only=True cannot unpickle.
        payload = load_checkpoint(resolved, map_location="cpu", trusted=True)
        if not isinstance(payload, dict):
            raise ValueError(
                f"{resolved} does not hold a checkpoint mapping; got {type(payload).__name__}"
            )

        quantization = str(payload.get("quantization", "fp32"))
        model_type = str(payload.get("modelType", "single"))

        if model_type == "composite":
            entries = list(payload.get("candidates") or [])
            if not entries:
                raise ValueError(
                    "A composite checkpoint must list its 'candidates'; without the "
                    "pool there is nothing for the router to choose between."
                )
            models = []
            for entry in entries:
                model = _build_route(entry, payload, config)
                state = entry.get("modelStateDict")
                if state is not None:
                    load_module_state(model, state, strict=True)
                models.append(model)
            router = _build_router(payload, [m.candidate_id for m in models], config)
            selector_fitted = router.selector_model is not None
        else:
            model = _build_route(payload.get("candidate") or {}, payload, config)
            state = payload.get("modelStateDict") or payload.get("model_state_dict")
            if state is not None:
                load_module_state(model, state, strict=True)
            models = [model]
            router = None
            selector_fitted = False

        return cls(
            models=models,
            router=router,
            device=device,
            checkpoint=resolved,
            config=config,
            quantization=quantization,
            selector_fitted=selector_fitted,
        )

    # -- inference ---------------------------------------------------------

    def predict(
        self,
        xray_bytes: bytes,
        mri_t2_bytes: Optional[bytes] = None,
        mri_r2_bytes: Optional[bytes] = None,
        *,
        want_heatmap: bool = True,
    ) -> Dict[str, Any]:
        """One case. Returns the predictor-level contract the server assembles."""

        import torch

        xray = self._decode_xray(xray_bytes).to(self.device)
        mri = None
        if mri_t2_bytes is not None and mri_r2_bytes is not None:
            mri = self._decode_mri(mri_t2_bytes, mri_r2_bytes).to(self.device)

        with torch.inference_mode():
            outputs = [model(xray, mri) for model in self.models]
            routed = self.router(outputs) if self.router is not None else outputs[0]

        probabilities = [float(value) for value in routed.probabilities[0].detach().cpu().tolist()]
        grade = int(routed.prediction[0].detach().cpu())
        executed_index = 0
        route_trace = routed.trace.route
        if route_trace is not None:
            executed_index = int(route_trace.selected[0].detach().cpu())

        heatmap = (
            self._grad_cam(self.models[executed_index], xray, mri)
            if want_heatmap
            else heatmap_lib.unavailable(GRAD_CAM_METHOD, "Saliency was not requested.")
        )

        return {
            "prediction": {
                "klGrade": grade,
                "classProbs": [round(value, 7) for value in probabilities],
                "confidence": round(max(probabilities), 7),
                "uncertainty": round(float(routed.uncertainty[0].detach().cpu()), 7),
            },
            "route": self._route_payload(routed, executed_index, mri is not None),
            "heatmap": heatmap,
            "runtime": RUNTIME_NAME,
            "quantization": self.quantization,
            "diagnostics": self._diagnostics(routed),
        }

    def _route_payload(self, routed: "Any", executed_index: int, mri_present: bool) -> Dict[str, Any]:
        trace = routed.trace.route
        route_id = self.models[executed_index].candidate_id
        switch_score = 0.0
        switch_flag = False
        if trace is not None:
            best = float(trace.best_alternative_score[0].detach().cpu())
            # A one-route pool scores its (nonexistent) best alternative at -inf.
            switch_score = best if best > -1e30 else 0.0
            switch_flag = bool(trace.switch_flag[0].detach().cpu())
        return {
            "routeId": route_id,
            "runtime": RUNTIME_NAME,
            "mriUsed": bool(routed.meta.mri_used and mri_present),
            "missingModalityFallback": bool(routed.meta.missing_modality_fallback),
            "switchScore": round(switch_score, 6),
            "switchFlag": switch_flag,
        }

    def _diagnostics(self, routed: "Any") -> Dict[str, Any]:
        """Reliability numbers a clinician's audit trail needs, scalarised.

        These are the deployment contract's own fields (posterior latent,
        measurement variance, Kalman gain, selector scores, route, fallback
        flag) reduced to per-request scalars. The full tensors are the ONNX
        signature's business; a JSON response reports their summaries.
        """

        payload: Dict[str, Any] = {
            "checkpoint": str(self.checkpoint),
            "device": str(self.device),
            "poolSize": len(self.models),
            "selectorFitted": self.selector_fitted,
        }
        rckf = routed.trace.rckf
        if rckf is not None:
            payload["measurementVariance"] = round(
                float(rckf.measurement_variance[0].mean().detach().cpu()), 6
            )
            payload["kalmanGain"] = round(
                float(rckf.kalman_gain[0].mean().detach().cpu()), 6
            )
            payload["rckfFallback"] = bool(rckf.missing_modality_fallback)
        route = routed.trace.route
        if route is not None:
            payload["routeIds"] = list(route.route_ids)
            payload["selectorScores"] = [
                round(float(value), 6) for value in route.scores[0].detach().cpu().tolist()
            ]
            payload["switchThreshold"] = route.threshold
            if not self.selector_fitted:
                payload["selectorNote"] = (
                    "No fitted selector in the checkpoint; scores are the "
                    "uncertainty-differential stand-in, not a trained routing policy."
                )
        return payload

    # -- preprocessing -----------------------------------------------------

    def _decode_xray(self, raw: bytes) -> "Any":
        """Upload bytes to the exact ``[1, 1, S, S]`` tensor the trainer feeds.

        The temporary file is not incidental: ``load_xray_image`` owns the crop,
        resize and CLAHE contract and takes a path, and going through it is what
        guarantees the served input matches the trained one.
        """

        import torch

        from koa_multimodal.data.xray import load_xray_image, normalize_xray_imagenet_gray

        if not raw:
            raise ValueError("Empty X-ray upload")
        with tempfile.TemporaryDirectory(prefix="koa_v7_xray_") as directory:
            source = Path(directory) / "upload.png"
            source.write_bytes(raw)
            image, _provenance = load_xray_image(
                source,
                size=self.config.input.xray_size,
                preprocessing_cfg=self.config.preprocessing,
            )
        # ImageNet-gray standardisation runs last, exactly as it does at the end
        # of KoaPairDataset.__getitem__. There is no augmentation at inference,
        # but the ordering is a property of the pipeline, not of the split.
        image = normalize_xray_imagenet_gray(image, self.config.preprocessing)
        return torch.from_numpy(image).unsqueeze(0).float()

    def _decode_mri(self, t2_bytes: bytes, r2_bytes: bytes) -> "Any":
        """Both NIfTI uploads to one ``[1, 2, D, H, W]`` tensor, channel order (T2, R2)."""

        import torch

        from koa_multimodal.data.mri import load_mri_pair

        if not t2_bytes or not r2_bytes:
            raise ValueError(
                "The fusion route consumes a two-channel (T2, R2) volume; a lone "
                "channel cannot form one."
            )
        with tempfile.TemporaryDirectory(prefix="koa_v7_mri_") as directory:
            t2_path = Path(directory) / "t2.nii.gz"
            r2_path = Path(directory) / "r2.nii.gz"
            t2_path.write_bytes(t2_bytes)
            r2_path.write_bytes(r2_bytes)
            volume = load_mri_pair(
                t2_path,
                r2_path,
                shape=self.config.input.mri_shape,
                preprocessing_cfg=self.config.preprocessing,
            )
        return torch.from_numpy(volume).unsqueeze(0).float()

    # -- explainability ----------------------------------------------------

    def _grad_cam(self, model: "Any", xray: "Any", mri: Optional["Any"]) -> Dict[str, Any]:
        """Grad-CAM over the X-ray encoder's last convolutional stage.

        The attribution method is Grad-CAM as published by Selvaraju et al.
        (2017) -- gradient-weighted class activation mapping -- reimplemented
        here against this model's ordinal head rather than taken from a library.

        Table 4.14 specifies this as an **FP32 shadow path**, and the
        wording is load-bearing. The served prediction has already been produced
        by the primary forward pass under whatever precision the deployed model
        uses; this is a *second*, gradient-enabled, unconditionally-FP32 pass
        whose only product is the map. Saliency therefore cannot perturb the
        grade, and INT8 rounding cannot corrupt the gradients the map is built
        from. ``mixed_precision`` is forced off for the duration and restored
        afterwards.

        Failure is reported, never raised: a transformer trunk whose only
        convolution is the patch-embedding stem, or a backward pass that yields
        no gradient, produces ``available: false`` with a reason the workbench
        prints. A demo that 500s because an overlay could not be drawn has traded
        a diagnosis for a decoration.
        """

        import torch
        from torch import nn

        target = _last_conv2d(model.xray_encoder)
        if target is None:
            return heatmap_lib.unavailable(
                GRAD_CAM_METHOD,
                "This X-ray encoder exposes no 2-D convolution to attribute against, "
                "so Grad-CAM has no spatial stage to read.",
            )

        captured: List["Any"] = []

        def _hook(_module: nn.Module, _inputs: Any, output: "Any") -> None:
            captured.append(output)

        handle = target.register_forward_hook(_hook)
        restore = getattr(model, "mixed_precision", False)
        try:
            model.mixed_precision = False  # FP32 shadow path
            with torch.enable_grad():
                output = model(xray.float(), None if mri is None else mri.float())
                score = output.probabilities[0, int(output.prediction[0])].clamp_min(1e-12).log()
                if not captured:
                    return heatmap_lib.unavailable(
                        GRAD_CAM_METHOD,
                        "The target convolution did not run in the shadow pass.",
                    )
                activations = captured[-1]
                if activations.ndim != 4:
                    return heatmap_lib.unavailable(
                        GRAD_CAM_METHOD,
                        f"Target activations are rank {activations.ndim}, not a "
                        "[B, C, H, W] feature map.",
                    )
                gradients = torch.autograd.grad(
                    score, activations, retain_graph=False, allow_unused=True
                )[0]
            if gradients is None:
                return heatmap_lib.unavailable(
                    GRAD_CAM_METHOD,
                    "The predicted-class score has no gradient path to the target "
                    "layer, so no attribution can be formed.",
                )
            cam = _cam_from(activations.detach(), gradients.detach())
            if cam is None:
                return heatmap_lib.unavailable(
                    GRAD_CAM_METHOD,
                    "Grad-CAM ran but every activation was suppressed by the ReLU, "
                    "leaving a uniformly zero map.",
                )
            field = _to_field(cam, heatmap_lib.HEATMAP_SIZE)
            payload = heatmap_lib.available(
                heatmap_lib.colourize(field), GRAD_CAM_METHOD, layers=["saliency"]
            )
            payload["targetLayer"] = _module_path(model.xray_encoder, target)
            payload["precision"] = "fp32_shadow_path"
            return payload
        finally:
            handle.remove()
            model.mixed_precision = restore


def _last_conv2d(encoder: "Any") -> Optional["Any"]:
    """The deepest 2-D convolution in the encoder, in definition order."""

    from torch import nn

    found = None
    for module in encoder.modules():
        if isinstance(module, nn.Conv2d):
            found = module
    return found


def _module_path(root: "Any", target: "Any") -> str:
    for name, module in root.named_modules():
        if module is target:
            return name or "<root>"
    return "<unknown>"


def _cam_from(activations: "Any", gradients: "Any") -> Optional["Any"]:
    """Channel weights from pooled gradients, then a ReLU'd weighted sum."""

    import torch

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    peak = float(cam.max())
    if peak <= 0.0:
        return None
    return cam / peak


def _to_field(cam: "Any", size: int) -> List[List[float]]:
    """Resample the CAM to the overlay grid and hand back plain floats."""

    from torch.nn import functional as F

    resized = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
    return [[float(value) for value in row] for row in resized[0, 0].cpu().tolist()]


# ------------------------------------------------------------------- singleton

_runtime: Optional[TorchRuntime] = None


def get_runtime(config: Optional[Config] = None) -> TorchRuntime:
    """The process-wide runtime, built on first use.

    Loading a checkpoint and moving it to the GPU costs seconds, so it happens
    once. The consequence is stated in :func:`runtime_manifest`: the environment
    is read at construction time, and changing it later requires a restart.
    """

    global _runtime
    if _runtime is None:
        _runtime = TorchRuntime.from_environment(config)
    return _runtime


def reset_runtime() -> None:
    """Drop the cached runtime. For tests; there is no endpoint that calls this."""

    global _runtime
    _runtime = None
