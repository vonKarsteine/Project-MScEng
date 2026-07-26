"""The only module in this package that calls ``torch.load``.

Two policies live here so they cannot drift apart across call sites:

* **Trust is explicit.** ``torch.load`` defaults to ``weights_only=True`` on
  torch >= 2.6 and ``False`` before it, so spelling the flag out at each call site
  means the package behaves differently depending on which torch is installed.
  :func:`load_checkpoint` pins ``weights_only=True`` and requires an explicit
  ``trusted=True`` to fall back -- checkpoints this package writes are trusted
  local pickles, but that has to be stated, not assumed.

* **Partial loads are loud.** ``load_state_dict(..., strict=False)`` reports what
  it skipped and callers routinely ignore it, so a checkpoint that supplies one
  tensor out of two hundred loads "successfully" and serves a randomly
  initialised model at full confidence. :func:`load_module_state` returns the
  counts and raises unless the caller states what it will tolerate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
from torch import nn

from koa_multimodal.core.errors import KoaError

PathLike = Union[str, Path]


def file_sha256(path: PathLike) -> str:
    """Stream a file through sha256. Used for checkpoint provenance in OOF artifacts."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(payload: Mapping[str, Any], path: PathLike) -> Path:
    """Write a checkpoint and return its path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), target)
    return target


def load_checkpoint(
    path: PathLike,
    *,
    map_location: Union[str, torch.device] = "cpu",
    trusted: bool = False,
    expect_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a checkpoint under an explicit trust policy.

    ``trusted=True`` is required for any checkpoint carrying non-tensor objects
    (this package's own, which embed feature specs and hyperparameters). Pass
    ``expect_sha256`` to bind the load to a recorded digest.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    if expect_sha256 is not None:
        actual = file_sha256(source)
        if actual != expect_sha256:
            raise KoaError(
                f"Checkpoint digest mismatch for {source}: "
                f"expected {expect_sha256}, found {actual}"
            )
    if trusted:
        return torch.load(source, map_location=map_location, weights_only=False)
    try:
        return torch.load(source, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001 - surfaced with a remedy, not swallowed
        raise KoaError(
            f"{source} could not be loaded with weights_only=True ({type(exc).__name__}). "
            "It carries non-tensor objects; pass trusted=True if you wrote it."
        ) from exc


@dataclass(frozen=True)
class StateLoadReport:
    """What actually landed when a state dict was applied."""

    matched: int
    missing: int
    unexpected: int
    total_target: int

    @property
    def matched_fraction(self) -> float:
        return self.matched / self.total_target if self.total_target else 0.0

    def describe(self) -> str:
        return (
            f"{self.matched}/{self.total_target} tensors matched "
            f"({self.missing} missing, {self.unexpected} unexpected)"
        )


def strip_prefix(state: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Select the keys under ``prefix`` and remove it. Empty dict when none match."""

    marker = prefix if prefix.endswith(".") else prefix + "."
    return {
        key[len(marker) :]: value for key, value in state.items() if key.startswith(marker)
    }


def load_module_state(
    module: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
    min_matched_fraction: float = 1.0,
) -> StateLoadReport:
    """Apply a state dict and refuse a silent partial load.

    With ``strict=True`` this is ``load_state_dict``'s own behaviour plus a report.
    With ``strict=False`` -- needed when grafting a branch encoder whose head is
    deliberately discarded -- the load must still cover
    ``min_matched_fraction`` of the target's tensors, so "nothing matched" and
    "almost nothing matched" both fail instead of passing quietly.
    """

    target_keys = set(module.state_dict().keys())
    result = module.load_state_dict(dict(state), strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    report = StateLoadReport(
        matched=len(target_keys) - len(missing),
        missing=len(missing),
        unexpected=len(unexpected),
        total_target=len(target_keys),
    )
    if report.matched_fraction < min_matched_fraction:
        raise KoaError(
            f"Refusing a partial checkpoint load into {type(module).__name__}: "
            f"{report.describe()}, below the required "
            f"{min_matched_fraction:.0%}. First missing keys: {missing[:5]}"
        )
    return report
