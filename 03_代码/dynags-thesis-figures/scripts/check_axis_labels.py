"""Flag known inconsistent axis terminology in figure sources."""

from __future__ import annotations

import ast
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from dynags_figures.labels import suggested_axis_label


def main() -> None:
    errors = []
    paths = sorted((PROJECT / "scripts").glob("figure_*.py")) + [PROJECT / "scripts/smoke_test.py"]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"set_xlabel", "set_ylabel", "supxlabel", "supylabel", "set_label"}:
                expressions = node.args[:1]
            elif node.func.attr == "text":
                expressions = node.args[2:3]
            else:
                continue
            for expression in expressions:
                for item in ast.walk(expression):
                    if isinstance(item, ast.Constant) and isinstance(item.value, str):
                        replacement = suggested_axis_label(item.value)
                        if replacement:
                            errors.append(f"{path.name}:{node.lineno}: {item.value!r} -> {replacement!r}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Axis terminology check passed for {len(paths)} figure/example scripts.")


if __name__ == "__main__":
    main()
