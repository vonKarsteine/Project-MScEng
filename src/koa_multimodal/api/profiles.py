"""The deployment profiles the workbench offers in its model dropdown.

A profile is a *deployment* choice, not a model class: it names which route the
demo should try, and which route it degrades to when the inputs cannot support
that choice. The frontend reads ``id``/``name``/``modalities`` (see
``frontend/src/utils/inference/index.js``); ``fallback`` and ``description`` are
extra and are ignored by a client that does not want them.

**Every ``fallback`` names a profile that exists in this list.** A fallback
pointing at an id nothing defines leaves the one field whose entire purpose is to
answer "what runs when MRI is missing" answering with something unresolvable, and
nothing about the dropdown looks wrong until a case actually degrades.
:func:`validate_profiles` runs at import, so that cannot happen silently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

#: The X-ray-only route every multimodal profile degrades to. Named once so the
#: fallback edges cannot drift apart from each other.
XRAY_FALLBACK_ID = "xray_cmodes_oof"

PROFILES: List[Dict[str, Any]] = [
    {
        "id": "fusion_cmodes_oof",
        "name": "Fusion C-MODES",
        "modalities": ["xray", "mri"],
        "fallback": XRAY_FALLBACK_ID,
        "description": (
            "The reported final architecture (Table 4.10, row 4): a "
            "four-route multimodal pool with the C-MODES risk-differential selector "
            "routing each case, calibrated out of fold."
        ),
    },
    {
        "id": "fusion_traditional_ensemble",
        "name": "Fusion OOF Stack",
        "modalities": ["xray", "mri"],
        "fallback": XRAY_FALLBACK_ID,
        "description": (
            "Static weighted posterior stacking over the same multimodal pool "
            "(Table 4.10, row 3). It never switches route, which is the baseline "
            "C-MODES is measured against."
        ),
    },
    {
        "id": "rckf_cv2_simsiam",
        "name": "RCKF ConvNeXt V2 + SimSiam SSL",
        "modalities": ["xray", "mri"],
        "fallback": XRAY_FALLBACK_ID,
        "description": (
            "The single strongest fusion route (Table 4.9, row 1) and the C-MODES "
            "default: residual-calibrated Kalman fusion of a ConvNeXt V2 radiograph "
            "prior with a SimSiam-pretrained volumetric measurement."
        ),
    },
    {
        "id": XRAY_FALLBACK_ID,
        "name": "X-ray C-MODES",
        "modalities": ["xray"],
        # The terminal node of the fallback graph: it needs no MRI, so there is
        # nothing for it to degrade to.
        "fallback": None,
        "description": (
            "Radiograph-only C-MODES routing (Table 4.10, row 2). Every multimodal "
            "profile falls back to this route when the T2 + R2 pair is incomplete."
        ),
    },
]

#: Keys every profile must carry, in the shape the frontend expects.
REQUIRED_KEYS = ("id", "name", "modalities", "fallback", "description")


def profile_ids() -> List[str]:
    return [str(profile["id"]) for profile in PROFILES]


def get_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    """The profile with this id, or ``None``. Never raises on unknown input."""

    for profile in PROFILES:
        if profile["id"] == profile_id:
            return profile
    return None


def resolve_profile_id(requested: Optional[str]) -> str:
    """Coerce a client-supplied profile id to one that exists.

    An unknown id is a client-side typo or a stale cached dropdown, not an
    attack; answering with the default profile keeps the demo usable and the
    response records which profile actually executed.
    """

    if requested and get_profile(requested) is not None:
        return str(requested)
    return str(PROFILES[0]["id"])


def fallback_for(profile_id: str) -> Optional[str]:
    """The profile this one degrades to when its modalities are unavailable."""

    profile = get_profile(profile_id)
    return None if profile is None else profile.get("fallback")


def validate_profiles() -> None:
    """Structural check, run at import: shape, unique ids, resolvable fallbacks."""

    seen = set()
    ids = set(profile_ids())
    for profile in PROFILES:
        missing = [key for key in REQUIRED_KEYS if key not in profile]
        if missing:
            raise ValueError(f"Profile {profile.get('id')!r} is missing {missing}")
        identifier = str(profile["id"])
        if identifier in seen:
            raise ValueError(f"Duplicate profile id {identifier!r}")
        seen.add(identifier)
        if not isinstance(profile["modalities"], list) or not profile["modalities"]:
            raise ValueError(f"Profile {identifier!r} must list at least one modality")
        fallback = profile["fallback"]
        if fallback is not None and fallback not in ids:
            raise ValueError(
                f"Profile {identifier!r} falls back to {fallback!r}, which is not a "
                "profile in this list. A fallback that cannot be resolved is worse "
                "than no fallback: it reads as a supported degradation path and is not one."
            )
        if fallback == identifier:
            raise ValueError(f"Profile {identifier!r} falls back to itself")


validate_profiles()
