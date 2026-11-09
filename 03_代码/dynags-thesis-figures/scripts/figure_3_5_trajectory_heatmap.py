"""Plot row-wise standardized mean trajectories for 24 maize traits."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_3_4_trait_trajectories import TIMES, TRAITS

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
PHENO_DIR = ROOT / "04_数据" / "真实玉米数据集" / "pheno"
TABLE = PROJECT / "data" / "figure_3_5_standardized_mean_trajectories.csv"
FIGURE = ROOT / "07_论文图表" / "3-5_候选表型标准化平均轨迹热图.png"


def calculate() -> tuple[np.ndarray, list[dict[str, str | int | float]]]:
    matrix = np.empty((len(TRAITS), len(TIMES)), dtype=float)
    rows: list[dict[str, str | int | float]] = []
    for row_index, trait in enumerate(TRAITS):
        frame = pd.read_csv(PHENO_DIR / f"{trait}.csv")
        if frame.shape != (400, len(TIMES)) or list(frame.columns) != list(map(str, TIMES)):
            raise ValueError(f"Unexpected shape or time columns: {trait}")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite value in {trait}")
        mean = values.mean(axis=0)
        center = float(mean.mean())
        spread = float(mean.std(ddof=0))
        if spread <= 0:
            raise ValueError(f"Constant mean trajectory: {trait}")
        standardized = (mean - center) / spread
        matrix[row_index] = standardized
        for time, value, z_score in zip(TIMES, mean, standardized, strict=True):
            rows.append(
                {
                    "trait": trait,
                    "time_coordinate": int(time),
                    "n_individuals": len(values),
                    "population_mean": float(value),
                    "row_mean": center,
                    "row_sd_ddof0": spread,
                    "z_score": float(z_score),
                    "source": f"pheno/{trait}.csv",
                }
            )
    return matrix, rows


def save_data(rows: list[dict[str, str | int | float]]) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(matrix: np.ndarray) -> None:
    limit = float(np.ceil(np.max(np.abs(matrix)) * 10) / 10)
    with thesis_style(base_font_size=8.0):
        plt.rcParams["axes.unicode_minus"] = True
        fig = plt.figure(figsize=(6.3, 7.3), dpi=300)
        ax = fig.add_axes((0.27, 0.12, 0.59, 0.80))
        image = ax.imshow(
            matrix, cmap="PuOr_r", vmin=-limit, vmax=limit,
            interpolation="nearest", aspect="auto", origin="upper",
        )
        ax.set_xticks(np.arange(len(TIMES)), labels=[str(t) for t in TIMES])
        ax.set_yticks(np.arange(len(TRAITS)), labels=TRAITS)
        ax.set_xlabel("Time coordinate", labelpad=10)
        ax.tick_params(axis="x", length=0, pad=6, labelsize=7.1)
        ax.tick_params(axis="y", length=0, pad=6, labelsize=7.5)
        ax.set_xticks(np.arange(-0.5, len(TIMES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(TRAITS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.65)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        color_axis = fig.add_axes((0.89, 0.12, 0.022, 0.80))
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.set_label("Z-score of mean trajectory", rotation=90, labelpad=10)
        colorbar.set_ticks([-2, -1, 0, 1, 2])
        colorbar.ax.tick_params(length=2, labelsize=7.3)
        colorbar.outline.set_linewidth(0.5)

        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


if __name__ == "__main__":
    values, data = calculate()
    save_data(data)
    draw(values)
    print(f"Traits: {len(TRAITS)}; time points: {len(TIMES)}; cells: {values.size}")
    print(f"Data: {TABLE}")
    print(f"Figure: {FIGURE}")
