"""Exception and warning types for the package.

Everything raised deliberately by this package derives from :class:`KoaError`, so
a caller can distinguish "this package rejected the input" from "torch blew up".
"""

from __future__ import annotations


class KoaError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(KoaError):
    """A configuration value is missing, unknown, or violates a stated invariant."""


class ContractError(KoaError):
    """A model output does not satisfy the :mod:`koa_multimodal.core.contract` contract.

    The commonest case: a CORN objective was handed a posterior-averaging route,
    which owns no ordinal head and therefore no threshold logits.
    """


class ProvenanceError(KoaError):
    """A prediction artifact fails the out-of-fold provenance gate.

    Raised rather than warned. Anything that trains a selector or a stacker must
    be able to prove its inputs are leak-free; a soft failure here would silently
    invalidate every downstream number.
    """


class DataLayoutError(KoaError):
    """The on-disk dataset does not match the index contract."""


class KoaWarning(UserWarning):
    """A recoverable degradation the caller should know about.

    Promoted to an error under pytest (see ``pyproject.toml``) so that a silent
    fallback cannot pass the suite unnoticed.
    """
