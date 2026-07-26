"""The layering rule, enforced mechanically rather than documented.

Imports point strictly downward through the layer graph, and every
``__init__.py`` stays empty. Together these keep the dependency graph acyclic and
make a package's cost predictable: importing the prediction-artifact reader must
not drag in the model zoo.

A syntax-only sweep such as ``compileall`` cannot catch either property. This
walks every module's AST instead, and runs in milliseconds.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator, Set

import pytest

import koa_multimodal

PACKAGE_ROOT = Path(koa_multimodal.__file__).resolve().parent

LAYERS = {
    "core": 0,
    "config": 1,
    "data": 2,
    "models": 2,
    "fusion": 3,
    "stats": 3,
    "ensemble": 4,
    "deploy": 5,
    "api": 5,
    "records": 5,
    "training": 6,
    "cli": 7,
}


def python_modules() -> Iterator[Path]:
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def layer_of(path: Path) -> int:
    parts = path.relative_to(PACKAGE_ROOT).parts
    if len(parts) == 1:  # koa_multimodal/__init__.py
        return -1
    return LAYERS[parts[0]]


def imported_packages(tree: ast.AST) -> Set[str]:
    """Sub-packages of koa_multimodal that this module imports."""

    found: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            name = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("koa_multimodal."):
                    found.add(alias.name.split(".")[1])
            continue
        else:
            continue
        if name.startswith("koa_multimodal."):
            found.add(name.split(".")[1])
    return found


def test_every_init_is_empty():
    """No re-exports anywhere.

    A populated ``__init__.py`` makes the import cost of a package unpredictable
    and gives one symbol several import paths. It also defeats the layer check
    below, which reads the layer off the import line.
    """

    populated = []
    for path in PACKAGE_ROOT.rglob("__init__.py"):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        has_code = any(
            not isinstance(node, (ast.Expr, ast.Assign))
            or (isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant))
            for node in tree.body
        )
        if has_code:
            populated.append(str(path.relative_to(PACKAGE_ROOT)))
    assert not populated, f"__init__.py must not re-export: {populated}"


def test_imports_point_strictly_downward():
    violations = []
    for path in python_modules():
        importer_layer = layer_of(path)
        if importer_layer < 0:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        importer_package = path.relative_to(PACKAGE_ROOT).parts[0]
        for package in imported_packages(tree):
            if package == importer_package:
                continue  # intra-package imports are fine
            target_layer = LAYERS.get(package)
            if target_layer is None:
                violations.append(f"{path.name}: unknown package {package!r}")
            elif target_layer >= importer_layer:
                violations.append(
                    f"{path.relative_to(PACKAGE_ROOT)} (L{importer_layer}) imports "
                    f"{package} (L{target_layer})"
                )
    assert not violations, "layering violations:\n  " + "\n  ".join(sorted(violations))


def test_no_bare_package_imports():
    """Every cross-package import names the module it depends on."""

    violations = []
    for path in python_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if parts[0] == "koa_multimodal" and len(parts) == 2:
                    violations.append(f"{path.name}: from {node.module} import ...")
    assert not violations, "bare package imports:\n  " + "\n  ".join(sorted(violations))


def test_no_sys_path_manipulation():
    """The package is installed; nothing may patch ``sys.path`` to find itself."""

    offenders = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in python_modules()
        if "sys.path.insert" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"sys.path manipulation found in {offenders}"


def test_torch_load_is_centralised():
    """``torch.load`` appears in exactly one module, which owns the trust policy."""

    callers = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in python_modules()
        if "torch.load(" in path.read_text(encoding="utf-8")
    ]
    assert callers == ["core/checkpoint.py"] or callers == [
        str(Path("core") / "checkpoint.py")
    ], f"torch.load must only be called from core/checkpoint.py, found in {callers}"
