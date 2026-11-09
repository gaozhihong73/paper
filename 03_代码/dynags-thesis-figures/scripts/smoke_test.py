"""Create a small data-driven PNG to verify the figure environment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dynags_figures import figure_size, save_figure, style_axis, thesis_style


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    thesis_root = project_root.parents[1]
    source = thesis_root / "04_数据" / "真实玉米数据集" / "pheno" / "bn.csv"
    output = project_root / "outputs" / "environment_smoke_test"

    data = pd.read_csv(source)
    time_points = [int(column) for column in data.columns]
    mean_trajectory = data.mean(axis=0).to_numpy()

    with thesis_style():
        fig, ax = plt.subplots(figsize=figure_size("column", 2.2))
        ax.plot(time_points, mean_trajectory, marker="o", color="#0072B2")
        ax.set_xlabel("Time point")
        ax.set_ylabel("Mean phenotype value")
        style_axis(ax, grid="y")
        paths = save_figure(fig, output)

    print(paths["png"])


if __name__ == "__main__":
    main()

