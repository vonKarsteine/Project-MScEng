"""Saliency overlays as base64 PNG data URLs, written with the standard library.

Both predictors return an overlay in the same shape and both build it here, so
the mock's synthetic blob and the runtime's real Grad-CAM composite over the
radiograph identically -- only the *values* and the ``method`` label differ.

The PNG encoder is ``zlib`` + ``struct`` rather than Pillow. The API is
stdlib-only by design (see :mod:`koa_multimodal.api.server`), and the mock
predictor in particular must run in an environment with no imaging stack at all,
so the one thing it needs from an image library is written out here in twenty
lines instead of pulling one in.

The palette matches the frontend's own mock overlay
(``frontend/src/utils/inference/mockAdapter.js``) stop for stop, so switching the
workbench between its client-side mock and this API does not change what the
clinician sees -- only where it came from.
"""

from __future__ import annotations

import base64
import math
import struct
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Matches ``OUTPUT_SIZE`` in ``frontend/src/components/AlignmentCanvas.jsx``.
#: The overlay is stretched over the on-screen crop region whatever size it is,
#: so this is fidelity rather than a hard requirement.
HEATMAP_SIZE = 384

#: ``(distance_from_peak, r, g, b, alpha)``. Ascending in distance: index 0 is the
#: hot centre, index -1 the fully transparent rim.
_PALETTE: Tuple[Tuple[float, int, int, int, float], ...] = (
    (0.00, 239, 68, 68, 0.92),
    (0.35, 245, 158, 11, 0.66),
    (0.65, 16, 185, 129, 0.38),
    (1.00, 37, 99, 235, 0.00),
)


# --------------------------------------------------------------------- encoding


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def encode_png_rgba(rows: Sequence[bytes], width: int, height: int) -> bytes:
    """Encode 8-bit RGBA scanlines as a PNG.

    ``rows`` holds ``height`` byte strings of ``4 * width`` bytes each, without
    per-row filter bytes -- filter type 0 (None) is prepended here, which is what
    lets the whole image be one ``zlib.compress`` call.
    """

    if len(rows) != height:
        raise ValueError(f"Expected {height} scanlines, got {len(rows)}")
    raw = b"".join(b"\x00" + row for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )


def png_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


# --------------------------------------------------------------------- colouring


def _colour(distance: float) -> Tuple[int, int, int, int]:
    """Piecewise-linear palette lookup. ``distance`` is 0 at the peak, 1 at the rim."""

    value = min(max(float(distance), 0.0), 1.0)
    for index in range(len(_PALETTE) - 1):
        low, low_r, low_g, low_b, low_a = _PALETTE[index]
        high, high_r, high_g, high_b, high_a = _PALETTE[index + 1]
        if value <= high:
            span = high - low
            t = 0.0 if span <= 0 else (value - low) / span
            return (
                int(round(low_r + (high_r - low_r) * t)),
                int(round(low_g + (high_g - low_g) * t)),
                int(round(low_b + (high_b - low_b) * t)),
                int(round(255.0 * (low_a + (high_a - low_a) * t))),
            )
    last = _PALETTE[-1]
    return (last[1], last[2], last[3], int(round(255.0 * last[4])))


#: 256-entry lookup table over ``distance``. Colouring a 384x384 overlay is
#: 147k lookups; interpolating each one in Python would dominate the response.
_LUT: Tuple[Tuple[int, int, int, int], ...] = tuple(
    _colour(index / 255.0) for index in range(256)
)


def colourize(field: Sequence[Sequence[float]]) -> bytes:
    """Render a 2-D activation field in ``[0, 1]`` as an RGBA PNG.

    ``1.0`` is the peak of the map and ``0.0`` is background, which the palette
    renders fully transparent -- so the overlay composites over the radiograph
    rather than hiding it.
    """

    height = len(field)
    if height == 0:
        raise ValueError("Activation field is empty")
    width = len(field[0])
    rows: List[bytes] = []
    for line in field:
        buffer = bytearray(width * 4)
        for x in range(width):
            intensity = line[x]
            if intensity < 0.0:
                intensity = 0.0
            elif intensity > 1.0:
                intensity = 1.0
            red, green, blue, alpha = _LUT[int((1.0 - intensity) * 255.0)]
            offset = x * 4
            buffer[offset] = red
            buffer[offset + 1] = green
            buffer[offset + 2] = blue
            buffer[offset + 3] = alpha
        rows.append(bytes(buffer))
    return encode_png_rgba(rows, width, height)


# ----------------------------------------------------------------- constructors


def gaussian_blob(
    centre_x: float,
    centre_y: float,
    radius: float,
    *,
    size: int = HEATMAP_SIZE,
) -> bytes:
    """A single soft Gaussian peak, as an RGBA PNG.

    ``centre_x``/``centre_y``/``radius`` are fractions of ``size``. The falloff is
    Gaussian rather than the linear ramp a CSS radial gradient produces, so the
    map looks like an activation map rather than a target reticle -- but it is
    still a closed-form blob and carries no information about the image. Whatever
    calls this must label it ``synthetic-demo``.
    """

    peak_x = centre_x * size
    peak_y = centre_y * size
    sigma = max(1e-3, 0.45 * radius * size)
    denominator = 2.0 * sigma * sigma
    field: List[List[float]] = []
    for y in range(size):
        dy2 = (y - peak_y) ** 2
        field.append([math.exp(-(dy2 + (x - peak_x) ** 2) / denominator) for x in range(size)])
    return colourize(field)


def unavailable(method: str, reason: str) -> Dict[str, Any]:
    """The negative form of the payload.

    ``dataUrl`` is present and ``None`` rather than absent: the frontend's
    ``normalizeHeatmap`` tolerates either, but a uniform key set means a consumer
    that indexes the dict directly gets ``None`` instead of a ``KeyError``.
    """

    return {"available": False, "dataUrl": None, "method": method, "reason": reason}


def available(
    png_bytes: bytes, method: str, *, layers: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """The positive form. ``method`` is what the workbench shows under 'Saliency'."""

    return {
        "available": True,
        "dataUrl": png_data_url(png_bytes),
        "method": method,
        "layers": list(layers) if layers else ["saliency"],
    }
