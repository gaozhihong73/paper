"""Draw all 24 longitudinal maize traits on their original numeric scales."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from dynags_figures.style import thesis_style
from dynags_figures.labels import AXIS_LABELS

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
PHENO_DIR = ROOT / "04_数据" / "真实玉米数据集" / "pheno"
TABLE = PROJECT / "data" / "figure_3_4_trait_trajectory_summary.csv"
FIGURE = ROOT / "07_论文图表" / "3-4_二十四个候选表型的纵向变化轨迹.png"
TIMES = np.array([10, 12, 14, 15, 17, 19, 24, 26, 37, 44, 51, 65, 80, 101])
TRAITS = [
    "bn", "CIVE", "COM", "ExB", "ExG", "ExGR", "ExR", "GLI",
    "GLI2", "gn", "GreenCoverage", "GRVI", "IKAW", "ints", "IPCA", "MGRVI",
    "NDI", "RGB_B", "RGB_G", "RGB_R", "RGBVI", "rn", "veg", "WI",
]


def summarize() -> list[dict[str, str | int | float]]:
    actual = {path.stem for path in PHENO_DIR.glob("*.csv")}
    if actual != set(TRAITS):
        raise ValueError(f"Trait list differs from the phenotype directory: {actual ^ set(TRAITS)}")
    rows: list[dict[str, str | int | float]] = []
    for trait in TRAITS:
        frame = pd.read_csv(PHENO_DIR / f"{trait}.csv")
        if frame.shape != (400, len(TIMES)) or list(frame.columns) != list(map(str, TIMES)):
            raise ValueError(f"Unexpected phenotype shape or time columns: {trait}")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing or nonfinite phenotype value: {trait}")
        mean = values.mean(axis=0)
        q25 = np.quantile(values, 0.25, axis=0)
        q75 = np.quantile(values, 0.75, axis=0)
        for time, center, lower, upper in zip(TIMES, mean, q25, q75, strict=True):
            rows.append(
                {
                    "trait": trait,
                    "time_coordinate": int(time),
                    "n_individuals": len(values),
                    "mean": float(center),
                    "q25": float(lower),
                    "q75": float(upper),
                    "source": f"pheno/{trait}.csv",
                }
            )
    return rows


def save_data(rows: list[dict[str, str | int | float]]) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(rows: list[dict[str, str | int | float]]) -> None:
    summaries = pd.DataFrame(rows)
    with thesis_style(base_font_size=7.3):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(6, 4, figsize=(6.3, 8.65), sharex=True)
        for index, (trait, ax) in enumerate(zip(TRAITS, axes.flat, strict=True)):
            subset = summaries.loc[summaries["trait"] == trait]
            time = subset["time_coordinate"].to_numpy(dtype=float)
            mean = subset["mean"].to_numpy(dtype=float)
            q25 = subset["q25"].to_numpy(dtype=float)
            q75 = subset["q75"].to_numpy(dtype=float)
            ax.fill_between(time, q25, q75, color="#A9C9DC", alpha=0.68,
                            linewidth=0, zorder=1)
            ax.plot(time, mean, color="#176B91", linewidth=1.25, zorder=2)
            span = max(float(q75.max() - q25.min()), 0.001)
            ax.set_ylim(float(q25.min() - 0.10 * span), float(q75.max() + 0.10 * span))
            ax.set_xlim(8, 103)
            ax.set_title(f"{chr(65 + index)}  {trait}", loc="left", pad=3,
                         fontsize=7.4, color="#173B51")
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(axis="both", labelsize=6.6, length=2, pad=1.5)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(axis="y", color="#DFE7EB", linewidth=0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#A7B8C0")
            ax.spines["bottom"].set_color("#A7B8C0")
            ax.set_xticks([10, 37, 65, 101])

        handles = [
            Line2D([], [], color="#176B91", lw=1.5, label="Mean (n = 400)"),
            Patch(facecolor="#A9C9DC", alpha=0.68, edgecolor="none",
                  label="25th–75th percentile"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.984),
                   ncol=2, fontsize=7.5, frameon=False)
        fig.supxlabel(AXIS_LABELS["time"], y=0.015, fontsize=8)
        fig.supylabel(AXIS_LABELS["phenotypic_value"], x=0.012, fontsize=8)
        fig.subplots_adjust(left=0.11, right=0.985, top=0.94, bottom=0.065,
                            wspace=0.48, hspace=0.72)
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=300, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    data = summarize()
    save_data(data)
    draw(data)
    print(f"Traits: {len(TRAITS)}; points: {len(data)}")
    print(f"Data: {TABLE}")
    print(f"Figure: {FIGURE}")
