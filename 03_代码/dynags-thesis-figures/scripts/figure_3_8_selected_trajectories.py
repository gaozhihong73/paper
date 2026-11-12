"""Plot individual trajectories for one high-H2 trait per provisional cluster."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_3_4_trait_trajectories import TIMES
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from dynags_figures.style import thesis_style
from dynags_figures.labels import AXIS_LABELS

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"
PHENO_DIR = ROOT / "04_数据" / "真实玉米数据集" / "pheno"
GROUP_SOURCE = DATA_DIR / "figure_3_7_cluster_assignments.csv"
H2_SOURCE = DATA_DIR / "figure_3_6_heritability_24_traits.csv"
SELECTION = DATA_DIR / "figure_3_8_representative_selection.csv"
SUMMARY = DATA_DIR / "figure_3_8_representative_mean_trajectories.csv"
FIGURE = ROOT / "07_论文图表" / "3-8_最终代表性状的个体纵向轨迹.png"


def select_traits() -> list[dict[str, str | int | float]]:
    groups = pd.read_csv(GROUP_SOURCE)
    heritability = pd.read_csv(H2_SOURCE)
    if groups.trait.nunique() != 24 or groups.cluster_k8.nunique() != 8:
        raise ValueError("Expected 24 traits and eight distinct provisional clusters")
    merged = groups.merge(heritability[["trait", "h2"]], on="trait", validate="one_to_one")
    if len(merged) != 24 or merged.h2.isna().any():
        raise ValueError("Cluster assignments and heritability values do not match")
    rows: list[dict[str, str | int | float]] = []
    cluster_order = merged.groupby("cluster_k8")["display_order"].min().sort_values().index
    for panel, cluster in enumerate(cluster_order, start=1):
        members = merged.loc[merged.cluster_k8 == cluster]
        chosen = members.sort_values(["h2", "display_order"], ascending=[False, True]).iloc[0]
        rows.append(
            {
                "panel": chr(64 + panel),
                "cluster_k8": int(cluster),
                "cluster_size": len(members),
                "selected_trait": str(chosen.trait),
                "selected_h2": float(chosen.h2),
                "selection_rule": "highest H2 within provisional trajectory cluster",
                "cluster_members": ";".join(members.sort_values("display_order").trait),
            }
        )
    return rows


def load_trajectories(
    selection: list[dict[str, str | int | float]],
) -> tuple[list[np.ndarray], list[dict[str, str | int | float]]]:
    arrays: list[np.ndarray] = []
    rows: list[dict[str, str | int | float]] = []
    for selected in selection:
        trait = str(selected["selected_trait"])
        frame = pd.read_csv(PHENO_DIR / f"{trait}.csv")
        if frame.shape != (400, len(TIMES)) or list(frame.columns) != list(map(str, TIMES)):
            raise ValueError(f"Unexpected phenotype dimensions or time points: {trait}")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {trait}")
        arrays.append(values)
        for time, mean in zip(TIMES, values.mean(axis=0), strict=True):
            rows.append(
                {
                    "trait": trait,
                    "time_coordinate": int(time),
                    "n_individuals": len(values),
                    "mean": float(mean),
                    "source": f"pheno/{trait}.csv",
                }
            )
    return arrays, rows


def save_data(
    selection: list[dict[str, str | int | float]],
    summary: list[dict[str, str | int | float]],
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, rows in [(SELECTION, selection), (SUMMARY, summary)]:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def draw(
    selection: list[dict[str, str | int | float]], arrays: list[np.ndarray]
) -> None:
    with thesis_style(base_font_size=8.0):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(4, 2, figsize=(6.3, 7.6), sharex=True, dpi=300)
        for selected, values, ax in zip(selection, arrays, axes.flat, strict=True):
            segments = np.stack(
                [np.broadcast_to(TIMES, values.shape), values], axis=-1
            )
            ax.add_collection(
                LineCollection(segments, colors="#7D9EAD", linewidths=0.43,
                               alpha=0.035, rasterized=True, zorder=1)
            )
            ax.plot(TIMES, values.mean(axis=0), color="#135C80", linewidth=1.8,
                    zorder=3)
            low, high = float(values.min()), float(values.max())
            span = max(float(high - low), 0.001)
            ax.set_ylim(float(low - 0.08 * span), float(high + 0.08 * span))
            ax.set_xlim(8, 103)
            ax.set_xticks([10, 24, 44, 65, 101])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.tick_params(axis="both", labelsize=7, length=2, pad=2)
            ax.grid(axis="y", color="#DFE7EB", linewidth=0.5)
            ax.set_title(f"{selected['panel']}  {selected['selected_trait']}",
                         loc="left", color="#173B51", fontsize=8.2, pad=4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#A7B8C0")
            ax.spines["bottom"].set_color("#A7B8C0")

        handles = [
            Line2D([], [], color="#7D9EAD", alpha=0.45, lw=0.8,
                   label="Individual trajectories (n = 400)"),
            Line2D([], [], color="#135C80", lw=1.8, label="Population mean"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
                   ncol=2, frameon=False, fontsize=7.5)
        fig.supxlabel(AXIS_LABELS["time"], y=0.015, fontsize=8)
        fig.supylabel(AXIS_LABELS["phenotypic_value"], x=0.012, fontsize=8)
        fig.subplots_adjust(left=0.12, right=0.985, top=0.925, bottom=0.075,
                            wspace=0.31, hspace=0.42)
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.13)
        plt.close(fig)


if __name__ == "__main__":
    chosen = select_traits()
    trajectories, means = load_trajectories(chosen)
    save_data(chosen, means)
    draw(chosen, trajectories)
    print("Selected:", ", ".join(str(row["selected_trait"]) for row in chosen))
    print(f"Selection: {SELECTION}")
    print(f"Figure: {FIGURE}")
