"""The three candidate-id namespaces, and the total conversions between them.

They are not interchangeable, and mixing them silently produces a lookup miss
rather than an error, so every conversion lives here:

============  ==========================================  =============================
namespace     example                                     produced by
============  ==========================================  =============================
base          ``xray_convnext_v2``, ``mri_simsiam_ssl``   the candidate catalogue
fusion        ``rckf_xray_convnext_v2_mri_simsiam_ssl``   :func:`fusion_candidate_id`
profile       ``rckf_cv2_simsiam``, ``fusion_cmodes_oof`` the deployment profile list
============  ==========================================  =============================
"""

from __future__ import annotations

import re
from typing import Tuple

from koa_multimodal.core.errors import KoaError

XRAY_PREFIX = "xray_"
MRI_PREFIX = "mri_"

#: The five fusion routes of §4.3.1-4.3.3.
FUSION_ROUTES = ("late_concat", "gated", "cross_attention", "contrastive", "rckf")

_FUSION_ID = re.compile(
    r"^(?P<route>" + "|".join(FUSION_ROUTES) + r")_(?P<xray>xray_.+?)_(?P<mri>mri_.+)$"
)


def is_xray_candidate(candidate_id: str) -> bool:
    return candidate_id.startswith(XRAY_PREFIX)


def is_mri_candidate(candidate_id: str) -> bool:
    return candidate_id.startswith(MRI_PREFIX)


def fusion_candidate_id(route: str, xray_candidate_id: str, mri_candidate_id: str) -> str:
    """Compose the long fusion id from a route and its two branch ids."""

    if route not in FUSION_ROUTES:
        raise KoaError(f"Unknown fusion route {route!r}; expected one of {FUSION_ROUTES}")
    if not is_xray_candidate(xray_candidate_id):
        raise KoaError(f"Not an X-ray candidate id: {xray_candidate_id!r}")
    if not is_mri_candidate(mri_candidate_id):
        raise KoaError(f"Not an MRI candidate id: {mri_candidate_id!r}")
    return f"{route}_{xray_candidate_id}_{mri_candidate_id}"


def split_fusion_candidate_id(candidate_id: str) -> Tuple[str, str, str]:
    """Inverse of :func:`fusion_candidate_id`: ``(route, xray_id, mri_id)``."""

    match = _FUSION_ID.match(candidate_id)
    if match is None:
        raise KoaError(f"Not a fusion candidate id: {candidate_id!r}")
    return match.group("route"), match.group("xray"), match.group("mri")


def is_fusion_candidate(candidate_id: str) -> bool:
    return _FUSION_ID.match(candidate_id) is not None


def modality_of(candidate_id: str) -> str:
    """``"xray"``, ``"mri"`` or ``"fusion"`` for any candidate id in any namespace."""

    if is_fusion_candidate(candidate_id):
        return "fusion"
    if is_xray_candidate(candidate_id):
        return "xray"
    if is_mri_candidate(candidate_id):
        return "mri"
    raise KoaError(f"Cannot infer a modality from candidate id {candidate_id!r}")
