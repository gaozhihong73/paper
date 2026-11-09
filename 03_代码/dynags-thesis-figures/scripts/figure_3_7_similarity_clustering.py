"""Plot mean-trajectory correlations and average-linkage clustering."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from figure_3_4_trait_trajectories import TIMES, TRAITS
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data" / "figure_3_5_standardized_mean_trajectories.csv"
DATA_DIR = PROJECT / "data"
FIGURE = ROOT / "07_论文图表" / "3-7_候选表型轨迹相似性及层次聚类结果.png"
GROUP_COUNT = 8


def calculate() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    data = pd.read_csv(SOURCE)
    if len(data) != len(TRAITS) * len(TIMES):
        raise ValueError("Incomplete standardized mean trajectories")
    matrix = data.pivot(index="trait", columns="time_coordinate", values="z_score")
    matrix = matrix.loc[TRAITS, TIMES]
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Missing or nonfinite trajectory value")
    values = matrix.to_numpy()
    if not np.allclose(values.mean(axis=1), 0, atol=1e-10):
        raise ValueError("Input trajectories are not row-centered")
    if not np.allclose(values.std(axis=1, ddof=0), 1, atol=1e-10):
        raise ValueError("Input trajectories are not row-standardized")

    correlation = np.clip(np.corrcoef(values), -1, 1)
    distance = np.maximum(1 - correlation, 0)
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    links = linkage(squareform(distance, checks=True), method="average",
                    optimal_ordering=True)
    tree = dendrogram(links, no_plot=True)
    order = list(tree["leaves"])
    groups = fcluster(links, t=GROUP_COUNT, criterion="maxclust")
    if len(set(groups)) != GROUP_COUNT:
        raise ValueError("The requested cluster count was not obtained")
    return correlation, distance, links, groups, order


def save_data(
    correlation: np.ndarray, links: np.ndarray, groups: np.ndarray, order: list[int]
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    correlation_path = DATA_DIR / "figure_3_7_trajectory_correlation.csv"
    with correlation_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trait", *TRAITS])
        for trait, values in zip(TRAITS, correlation, strict=True):
            writer.writerow([trait, *values])

    group_path = DATA_DIR / "figure_3_7_cluster_assignments.csv"
    with group_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["display_order", "trait", "cluster_k8"])
        for rank, original_index in enumerate(order, start=1):
            writer.writerow([rank, TRAITS[original_index], int(groups[original_index])])

    linkage_path = DATA_DIR / "figure_3_7_average_linkage.csv"
    with linkage_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left_id", "right_id", "distance_1_minus_r", "n_leaves"])
        writer.writerows(links)


def draw(
    correlation: np.ndarray, links: np.ndarray, groups: np.ndarray, order: list[int]
) -> None:
    ordered = correlation[np.ix_(order, order)]
    ordered_groups = groups[order]
    ordered_names = [TRAITS[index] for index in order]
    topology = links.copy()
    depths = np.zeros(2 * len(TRAITS) - 1)
    for index, (left, right, _, _) in enumerate(links):
        node = len(TRAITS) + index
        depths[node] = 1 + max(depths[int(left)], depths[int(right)])
        topology[index, 2] = depths[node]
    with thesis_style(base_font_size=7.5):
        plt.rcParams["axes.unicode_minus"] = True
        fig = plt.figure(figsize=(6.3, 7.1), dpi=300)
        heat_ax = fig.add_axes((0.27, 0.24, 0.53, 0.54))
        top_ax = fig.add_axes((0.27, 0.785, 0.53, 0.175))

        dendrogram(
            topology, ax=top_ax, no_labels=True, color_threshold=0,
            above_threshold_color="#415A68",
            link_color_func=lambda _: "#415A68",
        )
        for collection in top_ax.collections:
            collection.set_linewidth(0.85)
        top_ax.set_xlim(0, 10 * len(TRAITS))
        top_ax.set_ylim(0, float(topology[-1, 2]) * 1.05)
        top_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in top_ax.spines.values():
            spine.set_visible(False)
        fig.text(0.5, 0.98, "Average-linkage topology (branch lengths not to scale)",
                 ha="center", va="top", color="#526974", fontsize=7)

        image = heat_ax.imshow(ordered, cmap="RdBu_r", norm=Normalize(-1, 1),
                               interpolation="nearest", aspect="auto", origin="upper")
        heat_ax.set_xticks(np.arange(len(TRAITS)), labels=ordered_names, rotation=90)
        heat_ax.set_yticks(np.arange(len(TRAITS)), labels=ordered_names)
        heat_ax.yaxis.tick_left()
        heat_ax.tick_params(axis="x", length=0, pad=2, labelsize=6.6)
        heat_ax.tick_params(axis="y", length=0, pad=4, labelsize=6.6)
        for spine in heat_ax.spines.values():
            spine.set_visible(False)

        starts = np.r_[0, np.flatnonzero(np.diff(ordered_groups)) + 1]
        ends = np.r_[starts[1:], len(TRAITS)]
        for start, end in zip(starts, ends, strict=True):
            heat_ax.add_patch(
                Rectangle(
                    (start - 0.5, start - 0.5), end - start, end - start,
                    fill=False, edgecolor="#294B5C", linewidth=0.9,
                )
            )

        bar_ax = fig.add_axes((0.86, 0.24, 0.024, 0.54))
        bar = fig.colorbar(image, cax=bar_ax, orientation="vertical")
        bar.set_ticks([-1, -0.5, 0, 0.5, 1])
        bar.set_label("Pearson correlation (r)", rotation=90, labelpad=7)
        bar.ax.tick_params(length=2, labelsize=7)
        bar.outline.set_linewidth(0.5)

        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.14)
        plt.close(fig)


if __name__ == "__main__":
    matrix, distances, linkage_matrix, assignments, leaf_order = calculate()
    save_data(matrix, linkage_matrix, assignments, leaf_order)
    draw(matrix, linkage_matrix, assignments, leaf_order)
    print(f"Traits: {len(TRAITS)}; clusters: {len(set(assignments))}")
    print(f"Distance range: {distances.min():.3f}–{distances.max():.3f}")
    print(f"Figure: {FIGURE}")
