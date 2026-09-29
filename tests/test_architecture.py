"""
The layering, enforced.

These packages used to import each other in a cycle — converter -> scanner ->
scripts -> converter — which survived only because 33 of 36 internal imports
were written inside functions, where no tool could see them. Nothing warned;
it was only visible by reading every file. So the rule lives in a test now.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Each package may import from these, and nothing else in the project.
ALLOWED: dict[str, set[str]] = {
    "runtime":   set(),                                        # the bottom: depends on nothing
    "scanner":   {"runtime"},
    "speech":    {"runtime"},
    "converter": {"runtime", "scanner"},
    "validate":  {"runtime", "scanner", "converter", "speech"},
    "server":    {"runtime", "speech"},
    "tools":     {"runtime", "scanner", "converter", "speech", "validate"},
}
PACKAGES = set(ALLOWED)


def _imports(path: Path) -> set[str]:
    """Every project package this file imports, however the import is written."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
    return found & PACKAGES


def _files(package: str) -> list[Path]:
    return [p for p in (ROOT / package).rglob("*.py") if "__pycache__" not in p.parts]


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_package_only_imports_the_layers_below_it(package):
    for path in _files(package):
        for imported in _imports(path) - {package}:
            assert imported in ALLOWED[package], (
                f"{path.relative_to(ROOT)} imports '{imported}'. "
                f"{package} may only import {sorted(ALLOWED[package]) or 'nothing'} — "
                "adding this edge would reintroduce the cycle this layout removed."
            )


def test_the_declared_layering_has_no_cycles():
    """Guards the rules above, not just the code: ALLOWED itself must be a DAG."""
    def reaches(start, target, seen=frozenset()):
        if start in seen:
            return False
        return target in ALLOWED[start] or any(
            reaches(nxt, target, seen | {start}) for nxt in ALLOWED[start]
        )
    for package in PACKAGES:
        assert not reaches(package, package), f"{package} can reach itself"


def test_tools_is_a_leaf():
    """The old scripts/ package was a hub every layer imported from. tools/ is not."""
    for package in PACKAGES - {"tools"}:
        for path in _files(package):
            assert "tools" not in _imports(path), (
                f"{path.relative_to(ROOT)} imports from tools/. Command-line workflows "
                "are the top of the stack; anything shared belongs in a layer below."
            )


def test_no_package_reaches_into_another_packages_privates():
    """`from converter.quantize import _model_size_mb` — three of these existed."""
    pattern = re.compile(r"from (%s)[\w.]* import ([^\n(]+)" % "|".join(PACKAGES))
    for package in PACKAGES:
        for path in _files(package):
            for module, names in pattern.findall(path.read_text(encoding="utf-8")):
                if module == package:
                    continue                       # a package may use its own internals
                private = [n.strip() for n in names.split(",") if n.strip().startswith("_")]
                assert not private, (
                    f"{path.relative_to(ROOT)} imports {private} from {module}. "
                    "Cross-package means public: rename it or keep it inside."
                )
