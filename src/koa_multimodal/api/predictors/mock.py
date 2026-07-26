"""The deterministic mock predictor: a full response with no weights and no torch.

This checkout ships no trained weights, and the demo still has to be
demonstrable end to end -- profile selection, the alignment canvas, the class
posterior bars, the route panel, the saliency overlay and the feedback queue. So
the mock produces a *complete, well-formed* response rather than a stub, and it
is seeded by SHA-256 over ``(profile_id, filename)`` so the same upload always
yields the same answer. A demo that changes its mind between two runs of the same
case is worse than one that is obviously synthetic.

**It is never disguised as a model.** ``runtime`` is ``"mock"``, ``quantization``
is ``"mock"``, and the overlay is labelled ``method: "synthetic-demo"`` -- which is
the string the workbench prints under "Saliency", so the label is on screen and
not merely in the payload. The blob is a closed-form Gaussian whose position is
hashed from the filename; it is a function of the *name*, not of the image, and
carries no information about the anatomy underneath it.

The posterior shape mirrors ``frontend/src/utils/inference/mockAdapter.js`` --
an exponential ramp around a hashed centre -- so the API mock and the frontend's
own offline mock agree in behaviour. Only the hash differs (SHA-256 here, FNV-1a
there), so the two are not numerically identical and are not meant to be.

No torch, no numpy, no Pillow: this module imports the standard library and
:mod:`koa_multimodal.api.predictors.heatmap` only, which is what lets
``koa serve --check`` verify the response contract on a machine with nothing
installed.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional

from koa_multimodal.api.predictors import heatmap as heatmap_lib
from koa_multimodal.api.profiles import fallback_for, get_profile, resolve_profile_id

#: The five Kellgren-Lawrence grades.
NUM_CLASSES = 5

RUNTIME_NAME = "mock"

#: Shown by the workbench under "Saliency". The overlay must announce itself.
HEATMAP_METHOD = "synthetic-demo"


def _unit_hash(*parts: str) -> float:
    """A stable value in ``[0, 1)`` from the joined parts. SHA-256, so no collisions
    a filename could plausibly hit, and identical across processes and platforms --
    unlike ``hash()``, which is salted per interpreter run."""

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return (int(digest[:12], 16) % 100000) / 100000.0


def _posteriors(centre: float) -> List[float]:
    """A normalised exponential ramp peaked near ``centre``.

    Ordinal-looking on purpose: probability mass decays with distance from the
    peak grade, so the class-probability bars show the adjacent-grade ambiguity
    that KL grading actually has, rather than a random simplex point.
    """

    raw = [math.exp(-0.85 * abs(index - centre)) for index in range(NUM_CLASSES)]
    total = sum(raw)
    return [value / total for value in raw]


def _normalized_entropy(probabilities: List[float]) -> float:
    """Shannon entropy over ``log(K)``, so uncertainty is comparable across K.

    The same definition :func:`koa_multimodal.core.ordinal.normalized_entropy`
    applies to real outputs; duplicating four lines here is the price of the mock
    importing no tensor library.
    """

    entropy = -sum(value * math.log(max(value, 1e-8)) for value in probabilities)
    return entropy / math.log(NUM_CLASSES)


def mock_predict(
    *,
    profile_id: Optional[str] = None,
    xray_filename: Optional[str] = None,
    mri_paired: bool = False,
) -> Dict[str, Any]:
    """One deterministic inference.

    ``mri_paired`` is the frontend's own MRI-completeness predicate: a *complete*
    T2 + R2 pair, never "some MRI field was set". A lone channel cannot form the
    ``(2, D, H, W)`` volume the fusion routes consume, so it counts as missing
    here exactly as it does in the runtime.

    Returns the predictor-level contract described in
    :mod:`koa_multimodal.api.server`: ``prediction``, ``route``, ``heatmap``,
    ``runtime``, ``quantization`` and ``diagnostics``. The server adds the request
    id, the latency and the metadata envelope.
    """

    profile = resolve_profile_id(profile_id)
    seed = xray_filename or "xray"

    declared = get_profile(profile) or {}
    profile_wants_mri = "mri" in (declared.get("modalities") or [])
    mri_used = bool(mri_paired and profile_wants_mri)
    missing_modality_fallback = bool(profile_wants_mri and not mri_paired)
    # A multimodal profile without a complete pair executes its declared
    # fallback, and the response says so -- the workbench compares requested
    # against executed and flags the divergence.
    executed_profile = fallback_for(profile) if missing_modality_fallback else profile
    executed_profile = executed_profile or profile

    centre = 1.0 + 2.5 * _unit_hash(profile, seed) + (0.15 if mri_used else 0.0)
    probabilities = _posteriors(centre)
    grade = max(range(NUM_CLASSES), key=lambda index: probabilities[index])

    overlay = heatmap_lib.gaussian_blob(
        centre_x=0.36 + 0.28 * _unit_hash(seed, "cx"),
        centre_y=0.42 + 0.22 * _unit_hash(seed, "cy"),
        radius=0.24 + 0.12 * _unit_hash(seed, "r"),
    )

    return {
        "prediction": {
            "klGrade": grade,
            "classProbs": [round(value, 7) for value in probabilities],
            "confidence": round(max(probabilities), 7),
            "uncertainty": round(_normalized_entropy(probabilities), 7),
        },
        "route": {
            "routeId": executed_profile,
            "runtime": RUNTIME_NAME,
            "mriUsed": mri_used,
            "missingModalityFallback": missing_modality_fallback,
            "switchScore": round(_unit_hash(profile, seed, "switch") * 0.18, 3),
            "switchFlag": _unit_hash(profile, seed, "flag") > 0.5,
        },
        "heatmap": heatmap_lib.available(overlay, HEATMAP_METHOD),
        "runtime": RUNTIME_NAME,
        "quantization": RUNTIME_NAME,
        "diagnostics": {
            "seededBy": "sha256(profileId|filename)",
            "note": (
                "Deterministic synthetic response. No trained weights were loaded "
                "and no image content was read; the saliency overlay is a closed-form "
                "Gaussian positioned from the filename hash."
            ),
        },
    }
