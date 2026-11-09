"""Draw Figure 3-2 from the original PLINK quality-control logs."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "04_数据" / "真实玉米数据集" / "plink"
FIGURE_DIR = ROOT / "07_论文图表"
DATA_DIR = PROJECT / "data"
OUTPUT_NAME = "3-2_SNP质量控制过程中的位点数量变化.png"


def find_number(pattern: str, content: str, source: Path) -> int:
    match = re.search(pattern, content, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Expected count not found in {source.name}: {pattern}")
    return int(match.group(1))


def load_counts() -> list[dict[str, str | int]]:
    qc_path = LOG_DIR / "data_qc_strict.log"
    ld_path = LOG_DIR / "ld_pruned.log"
    qc = qc_path.read_text(encoding="utf-8")
    ld = ld_path.read_text(encoding="utf-8")

    raw = find_number(r"--vcf: (\d+) variants scanned", qc, qc_path)
    type_removed = find_number(r"(\d+) excluded by\s*\n--max-alleles", qc, qc_path)
    type_retained = find_number(r"--max-alleles \+ --snps-only, (\d+) remaining", qc, qc_path)
    missing_removed = find_number(r"--geno: (\d+) variants removed", qc, qc_path)
    hwe_removed = find_number(r"--hwe: (\d+) variants removed", qc, qc_path)
    maf_removed = find_number(r"(\d+) variants removed due to allele frequency", qc, qc_path)
    qc_retained = find_number(r"(\d+) variants remaining after main filters", qc, qc_path)
    ld_loaded = find_number(r"(\d+) variants loaded from data_qc_strict.pvar", ld, ld_path)
    ld_removed = find_number(r"--indep-pairwise.*?: (\d+)/\d+ variants removed", ld, ld_path)

    counts = [raw, type_retained]
    removals = [type_removed, missing_removed, hwe_removed, maf_removed, ld_removed]
    for removed in removals[1:]:
        counts.append(counts[-1] - removed)
    if raw - type_removed != type_retained:
        raise ValueError("Variant-type filtering counts do not reconcile")
    if counts[-2] != qc_retained or qc_retained != ld_loaded:
        raise ValueError("QC and LD pruning logs do not reconcile")
    if counts[-1] <= 0:
        raise ValueError("LD pruning produced a nonpositive variant count")

    labels = [
        ("Raw\nvariants", "VCF input"),
        ("Biallelic\nSNPs", "A/C/G/T"),
        ("Missingness\nfilter", "Missing ≤ 5%"),
        ("HWE\nfilter", r"p ≥ $10^{-6}$"),
        ("MAF\nfilter", "MAF ≥ 5%"),
        ("LD\npruning", "100 kb; r² = 0.2"),
    ]
    stages = []
    for index, ((label, condition), retained) in enumerate(zip(labels, counts, strict=True)):
        stages.append(
            {
                "stage": index + 1,
                "label": label.replace("\n", " "),
                "condition": condition,
                "retained": retained,
                "removed": 0 if index == 0 else removals[index - 1],
                "source": qc_path.name if index < 5 else ld_path.name,
            }
        )
    return stages


def save_data(stages: list[dict[str, str | int]]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "figure_3_2_qc_counts.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stages[0]))
        writer.writeheader()
        writer.writerows(stages)
    return path


def draw(stages: list[dict[str, str | int]]) -> Path:
    with thesis_style(base_font_size=8.0):
        plt.rcParams["axes.unicode_minus"] = True
        fig, ax = plt.subplots(figsize=(6.3, 4.1), dpi=300)
        counts = np.array([int(stage["retained"]) for stage in stages])
        positions = np.arange(len(stages) - 1, -1, -1)
        labels = [f"{stage['label']}\n{stage['condition']}" for stage in stages]
        # Marker area follows the logarithm of the retained count, matching the x-axis.
        log_counts = np.log10(counts)
        marker_sizes = 30 + 52 * (log_counts - log_counts.min()) / np.ptp(log_counts)

        ax.plot(counts, positions, color="#91B4C6", linewidth=1.5, zorder=2)
        ax.scatter(counts[:-1], positions[:-1], s=marker_sizes[:-1], color="#176B91",
                   edgecolor="white", linewidth=0.55, zorder=3)
        ax.scatter(counts[-1], positions[-1], s=marker_sizes[-1], color="#D55E00",
                   edgecolor="white", linewidth=0.65, zorder=4)

        for index, (count, position) in enumerate(zip(counts, positions, strict=True)):
            color = "#9F4700" if index == len(stages) - 1 else "#193C51"
            ax.annotate(f"{count:,}", (count, position), xytext=(8, 0),
                        textcoords="offset points", ha="left", va="center",
                        color=color, fontsize=8.0, fontweight="bold")
            if index:
                previous = counts[index - 1]
                percentage = 100 * int(stages[index]["removed"]) / previous
                ax.annotate(
                    f"{percentage:.1f}% removed",
                    (np.sqrt(previous * count), (positions[index - 1] + position) / 2),
                    xytext=(0, 10), textcoords="offset points", ha="center", va="bottom",
                    color="#526974", fontsize=7.0,
                )

        ax.set_xscale("log")
        ax.set_xlim(1.7e6, 7e8)
        ax.set_ylim(-0.4, len(stages) - 0.6)
        ax.set_xticks([1e7, 1e8], labels=["10 million", "100 million"])
        ax.minorticks_off()
        ax.set_yticks(positions, labels=labels)
        ax.set_xlabel("Retained variants (log scale)", labelpad=9)
        ax.tick_params(axis="x", labelsize=7.7, length=3)
        ax.tick_params(axis="y", labelsize=7.7, length=0, pad=8)
        ax.grid(axis="x", color="#DAE5EB", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#A6B8C2")
        fig.subplots_adjust(left=0.31, right=0.97, top=0.97, bottom=0.18)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        output = FIGURE_DIR / OUTPUT_NAME
        fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.13)
        plt.close(fig)
        return output


if __name__ == "__main__":
    values = load_counts()
    data_path = save_data(values)
    figure_path = draw(values)
    print(f"Data: {data_path}")
    print(f"Figure: {figure_path}")
