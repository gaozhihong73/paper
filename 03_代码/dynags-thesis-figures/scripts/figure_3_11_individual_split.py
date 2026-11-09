"""Visualize the saved individual split of simulation replicate 08.

Each tile represents one individual's genotype and complete 20-point trajectory.
The plot uses actual saved split indices, not an illustrative random assignment.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "04_数据" / "3-4_P10000_NTR1600_H0.8_rep_08"
RUN = ROOT / "04_数据" / "3-6_P10000_NTR1600_H0.8_rep_08"
SPLIT = SOURCE / "splits" / "split_seed73.npz"
DATA = PROJECT / "data" / "figure_3_11_individual_split.csv"
FIGURE = ROOT / "07_论文图表" / "3-11_模拟数据集的个体级划分及样本分配.png"

SPLITS = ("Training", "Validation", "Test")
COLORS = ("#0072B2", "#E69F00", "#009E73")
INK = "#193B50"
MUTED = "#506A7A"


def load_split() -> tuple[list[str], np.ndarray, dict]:
    manifest = json.loads((SOURCE / "data_manifest.json").read_text(encoding="utf-8"))
    record = json.loads((RUN / "preprocessing" / "preprocessing.json").read_text(
        encoding="utf-8"
    ))
    split_hash = hashlib.sha256(SPLIT.read_bytes()).hexdigest()
    if split_hash != record["split_sha256"]:
        raise ValueError("Saved split differs from the model run's recorded split")
    if hashlib.sha256((RUN / "preprocessing" / "split.npz").read_bytes()).hexdigest() != split_hash:
        raise ValueError("The model run's split copy differs from the source")

    with SPLIT.open("rb") as handle, np.load(handle) as saved:
        indices = {name: saved[f"{name}_idx"].astype(int) for name in ("train", "val", "test")}
        metadata = json.loads(str(saved["metadata"]))

    ids = pd.read_csv(SOURCE / "sample_ids.csv", dtype=str)["SampleID"].tolist()
    pheno = pd.read_csv(SOURCE / "sim_pheno.csv")
    genotype = np.load(SOURCE / "sim_snp.npy", mmap_mode="r")
    n = len(ids)
    if len(set(ids)) != n or pheno["SampleID"].tolist() != ids:
        raise ValueError("Genotype/phenotype sample order cannot be verified")
    if metadata["sample_ids"] != ids or n != manifest["genotype_shape"][0]:
        raise ValueError("Sample IDs differ from split metadata or manifest")
    if genotype.shape != tuple(manifest["genotype_shape"]):
        raise ValueError("Genotype shape differs from the data manifest")
    expected_pheno_columns = ["SampleID", *manifest["phenotype_columns"]]
    if (pheno.shape != (n, 21) or manifest["phenotype_shape"] != [n, 20]
            or pheno.columns.tolist() != expected_pheno_columns):
        raise ValueError("Expected one phenotype with 20 time points")
    if metadata["seed"] != 73 or metadata["n_samples"] != n:
        raise ValueError("Unexpected split seed or sample count")

    assignment = np.full(n, -1, dtype=np.int8)
    counts = {}
    for code, (key, label) in enumerate(zip(("train", "val", "test"), SPLITS, strict=True)):
        idx = indices[key]
        if idx.ndim != 1 or (idx < 0).any() or (idx >= n).any():
            raise ValueError(f"Invalid {key} indices")
        if np.unique(idx).size != idx.size or (assignment[idx] != -1).any():
            raise ValueError("Split indices repeat or overlap")
        assignment[idx] = code
        counts[label] = int(idx.size)
    if (assignment < 0).any() or counts != {"Training": 1600, "Validation": 200, "Test": 200}:
        raise ValueError("Split does not cover 2000 individuals with expected counts")
    if metadata["ratios"] != [0.8, 0.1, 0.1]:
        raise ValueError("Split ratios differ from the saved protocol")
    return ids, assignment, counts


def write_plot_data(ids: list[str], assignment: np.ndarray) -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with DATA.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["individual_order", "sample_id", "subset"])
        writer.writerows(
            (order, sample_id, SPLITS[int(code)])
            for order, (sample_id, code) in enumerate(zip(ids, assignment, strict=True), 1)
        )


def draw(assignment: np.ndarray, counts: dict) -> None:
    grid = assignment.reshape(40, 50)
    with thesis_style(base_font_size=8.0):
        fig = plt.figure(figsize=(6.3, 3.55), dpi=300)
        ax = fig.add_axes((0.14, 0.16, 0.52, 0.72))
        ax.imshow(grid, cmap=ListedColormap(COLORS), interpolation="nearest",
                  vmin=-0.5, vmax=2.5, aspect="equal", origin="upper")
        ax.set_xticks([0, 9, 19, 29, 39, 49], ["1", "10", "20", "30", "40", "50"])
        ax.set_yticks([0, 9, 19, 29, 39],
                      ["1–50", "451–500", "951–1000", "1451–1500", "1951–2000"])
        ax.set_xlabel("Position within 50-individual block", labelpad=5)
        ax.set_ylabel("Sample ID range", labelpad=5)
        ax.tick_params(length=2, color="#8BA3B0", pad=3)
        for spine in ax.spines.values():
            spine.set_edgecolor("#506A7A")
            spine.set_linewidth(0.7)

        fig.text(0.70, 0.80, "Subset", color=INK, fontsize=8.6, weight="bold")
        for row, (label, color) in enumerate(zip(SPLITS, COLORS, strict=True)):
            y = 0.66 - row * 0.18
            fig.patches.append(
                Rectangle((0.70, y - 0.012), 0.026, 0.046,
                          transform=fig.transFigure, color=color, linewidth=0)
            )
            fig.text(0.742, y + 0.028, label, color=INK, fontsize=8.5,
                     weight="bold", va="center")
            fraction = counts[label] / assignment.size
            fig.text(0.742, y - 0.026, f"{counts[label]:,} individuals  ({fraction:.0%})",
                     color=MUTED, fontsize=7.4, va="center")
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=300, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    sample_ids, split_assignment, split_counts = load_split()
    write_plot_data(sample_ids, split_assignment)
    draw(split_assignment, split_counts)
    print(FIGURE)
