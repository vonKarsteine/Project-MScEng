"""Load, override and validate the typed configuration.

The TOML parser is ``tomllib`` on Python 3.11+ and ``tomli`` below it. They are the
same parser -- ``tomllib`` was adopted into the standard library *from* ``tomli``
-- so parsing is identical across interpreters, and inline tables, multi-line
arrays and dotted section names all round-trip.

Precedence, lowest to highest: dataclass defaults, the TOML file, then
``KOA_<SECTION>__<KEY>`` environment variables.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10
    import tomli as tomllib  # type: ignore[no-redef]

from koa_multimodal.config.paths import default_config_path
from koa_multimodal.config.schema import Config
from koa_multimodal.core.errors import ConfigError

PathLike = Union[str, Path]

ENV_PREFIX = "KOA_"
ENV_SEPARATOR = "__"


# ------------------------------------------------------------------ reflection


def _is_config_dataclass(annotation_type: Any) -> bool:
    return dataclasses.is_dataclass(annotation_type) and isinstance(annotation_type, type)


def field_paths(node_type: Any = Config, prefix: Tuple[str, ...] = ()) -> Iterator[Tuple[str, ...]]:
    """Every leaf field path in the config tree, e.g. ``("rckf", "prior_variance")``.

    Reflective, so the bijection and consumer tests never go stale as fields move.
    """

    for spec in dataclasses.fields(node_type):
        child = spec.type if isinstance(spec.type, type) else _resolve_type(node_type, spec)
        if _is_config_dataclass(child):
            yield from field_paths(child, prefix + (spec.name,))
        else:
            yield prefix + (spec.name,)


def _resolve_type(owner: Any, spec: "dataclasses.Field") -> Any:
    """Resolve a string annotation left behind by ``from __future__ import annotations``."""

    import sys

    module = sys.modules[owner.__module__]
    try:
        return eval(spec.type, vars(module))  # noqa: S307 - our own module namespace
    except Exception:  # noqa: BLE001 - a non-dataclass annotation is a leaf, which is fine
        return spec.type


def leaf_keys(mapping: Mapping[str, Any], prefix: Tuple[str, ...] = ()) -> Iterator[Tuple[str, ...]]:
    """Every leaf key path in a parsed TOML mapping."""

    for key, value in mapping.items():
        if isinstance(value, dict):
            yield from leaf_keys(value, prefix + (key,))
        else:
            yield prefix + (key,)


# --------------------------------------------------------------------- loading


def _unknown_keys(raw: Mapping[str, Any]) -> List[str]:
    known = {".".join(path) for path in field_paths()}
    return sorted({".".join(path) for path in leaf_keys(raw)} - known)


def _coerce(value: Any, target: Any) -> Any:
    """Coerce a TOML scalar or array into the field's declared shape."""

    origin = getattr(target, "__origin__", None)
    if origin is tuple or target is tuple:
        return tuple(value)
    if target is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _build(node_type: Any, raw: Mapping[str, Any]) -> Any:
    kwargs: Dict[str, Any] = {}
    for spec in dataclasses.fields(node_type):
        if spec.name not in raw:
            continue
        child = _resolve_type(node_type, spec)
        value = raw[spec.name]
        if _is_config_dataclass(child):
            if not isinstance(value, dict):
                raise ConfigError(f"Section {spec.name!r} must be a table")
            kwargs[spec.name] = _build(child, value)
        else:
            kwargs[spec.name] = _coerce(value, child)
    return node_type(**kwargs)


def _parse_env_scalar(text: str) -> Any:
    lowered = text.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _apply_env_overrides(raw: Dict[str, Any], environ: Mapping[str, str]) -> Dict[str, Any]:
    """``KOA_RCKF__PRIOR_VARIANCE=0.2`` overrides ``[rckf] prior_variance``."""

    known = {".".join(path).lower(): path for path in field_paths()}
    for name, value in environ.items():
        if not name.startswith(ENV_PREFIX) or ENV_SEPARATOR not in name:
            continue
        dotted = name[len(ENV_PREFIX) :].replace(ENV_SEPARATOR, ".").lower()
        path = known.get(dotted)
        if path is None:
            continue
        cursor: Dict[str, Any] = raw
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[path[-1]] = _parse_env_scalar(value)
    return raw


def load_config(
    path: Optional[PathLike] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    apply_env: bool = True,
) -> Config:
    """Read ``configs/default.toml`` (or ``path``) into a validated :class:`Config`."""

    source = Path(path) if path is not None else default_config_path()
    if not source.is_file():
        raise ConfigError(f"Configuration file does not exist: {source}")
    raw = tomllib.loads(source.read_text(encoding="utf-8"))

    unknown = _unknown_keys(raw)
    if unknown:
        raise ConfigError(
            f"Unknown configuration keys in {source.name}: {unknown}. "
            "A key with no field is a typo or a leftover; both would otherwise be "
            "ignored silently."
        )
    if apply_env:
        raw = _apply_env_overrides(raw, environ if environ is not None else os.environ)
    return _build(Config, raw).validate()


def config_to_mapping(config: Any = None) -> Dict[str, Any]:
    """Round-trip a :class:`Config` back to a nested plain mapping."""

    return dataclasses.asdict(config if config is not None else Config())
