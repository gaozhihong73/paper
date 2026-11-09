"""Plot supplied heritability estimates for the 24 candidate maize traits."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_3_4_trait_trajectories import TRAITS

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "04_数据" / "真实玉米数据集" / "all_phenotype_h2.csv"
TABLE = PROJECT / "data" / "figure_3_6_heritability_24_traits.csv"
FIGURE = ROOT / "07_论文图表" / "3-6_二十四个候选表型的遗传力分布.png"


def load_estimates() -> list[dict[str, str | int | float]]:
    frame = pd.read_csv(SOURCE)
    if "H2" not in frame.columns or "Unnamed: 0" not in frame.columns:
        raise ValueError("Unexpected heritability file columns")
    traits = frame["Unnamed: 0"].astype(str)
    if traits.duplicated().any() or set(traits) != set(TRAITS) | {"VARI"}:
        raise ValueError("Heritability table does not match the 24 candidate traits plus VARI")
    subset = frame.loc[traits.isin(TRAITS), ["Unnamed: 0", "H2"]].copy()
    if not np.isfinite(subset["H2"]).all() or not subset["H2"].between(0, 1).all():
        raise ValueError("Heritability estimates must be finite and between 0 and 1")
    subset = subset.sort_values("H2", ascending=False, kind="stable").reset_index(drop=True)
    return [
        {
            "rank": index + 1,
            "trait": row["Unnamed: 0"],
            "h2": float(row["H2"]),
            "source": "all_phenotype_h2.csv",
        }
        for index, row in subset.iterrows()
    ]


def save_data(rows: list[dict[str, str | int | float]]) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(rows: list[dict[str, str | int | float]]) -> None:
    values = np.array([float(row["h2"]) for row in rows])
    positions = np.arange(len(rows))
    labels = [str(row["trait"]) for row in rows]
    with thesis_style(base_font_size=8.0):
        fig, ax = plt.subplots(figsize=(6.3, 7.2), dpi=300)
        ax.hlines(positions, 0, values, color="#9BBBCB", linewidth=1.2, zorder=2)
        ax.scatter(values, positions, s=26, color="#176B91", edgecolor="white",
                   linewidth=0.5, zorder=3)
        for index, value in enumerate(values):
            ax.text(value + 0.022, index, f"{value:.3f}", va="center", ha="left",
                    color="#354C59", fontsize=7.3)

        ax.set_yticks(positions, labels=labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("Heritability estimate (H2)", labelpad=8)
        ax.tick_params(axis="y", length=0, pad=7, labelsize=7.9)
        ax.tick_params(axis="x", length=3, labelsize=7.5)
        ax.grid(axis="x", color="#DDE6EB", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#9FB3BF")
        fig.subplots_adjust(left=0.27, right=0.93, top=0.98, bottom=0.105)

        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)


if __name__ == "__main__":
    data = load_estimates()
    save_data(data)
    draw(data)
    print(f"Traits: {len(data)}; H2 range: {data[-1]['h2']:.3f}–{data[0]['h2']:.3f}")
    print(f"Data: {TABLE}")
    print(f"Figure: {FIGURE}")
