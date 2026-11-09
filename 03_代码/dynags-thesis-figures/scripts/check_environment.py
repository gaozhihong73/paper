"""Verify that the isolated figure environment can import its core packages."""

from __future__ import annotations

import importlib
from pathlib import Path

PACKAGES = (
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "PIL",
    "pycirclize",
    "pyreadr",
)


def main() -> None:
    for name in PACKAGES:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "installed")
        print(f"{name}: {version}")

    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root.parents[1] / "04_数据" / "真实玉米数据集"
    output_root = project_root.parents[1] / "07_论文图表"
    print(f"data_root_exists: {data_root.exists()}")
    print(f"output_root_exists: {output_root.exists()}")


if __name__ == "__main__":
    main()

