"""Plot the chromosome-wide distribution of LD-pruned SNPs in 1 Mb bins."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyreadr
from matplotlib.colors import Normalize

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "04_数据" / "真实玉米数据集" / "drawing" / "data" / "pos_info.RData"
LOG = ROOT / "04_数据" / "真实玉米数据集" / "plink" / "data_ld_pruned.log"
FIGURE = ROOT / "07_论文图表" / "3-3_LD剪枝后SNP的全基因组分布.png"
TABLE = PROJECT / "data" / "figure_3_3_snp_density_1mb.csv"
BIN_SIZE = 1_000_000


def load_density() -> list[dict[str, int]]:
    frame = pyreadr.read_r(str(SOURCE))["pos_info"]
    if list(frame.columns) != ["chr", "pos"] or frame.isna().any().any():
        raise ValueError("Unexpected chromosome-position table")
    if len(frame) != 2_860_770 or set(frame["chr"]) != set(range(1, 11)):
        raise ValueError("Unexpected SNP total or chromosome set")
    if (frame["pos"] <= 0).any() or frame.duplicated().any():
        raise ValueError("Invalid or duplicate chromosome-position records")
    if "2860770 variants remaining after main filters" not in LOG.read_text(encoding="utf-8"):
        raise ValueError("Position table does not agree with PLINK output log")

    rows: list[dict[str, int]] = []
    for chromosome, subset in frame.groupby("chr", sort=True):
        positions = subset["pos"].to_numpy()
        maximum = int(positions.max())
        counts = np.bincount(positions // BIN_SIZE, minlength=maximum // BIN_SIZE + 1)
        if int(counts.sum()) != len(subset):
            raise ValueError(f"Chromosome {chromosome} counts do not reconcile")
        for index, count in enumerate(counts):
            rows.append(
                {
                    "chromosome": int(chromosome),
                    "window_start_bp": index * BIN_SIZE,
                    "window_end_bp": (index + 1) * BIN_SIZE,
                    "snp_count": int(count),
                    "observed_max_bp": maximum,
                }
            )
    if sum(row["snp_count"] for row in rows) != len(frame):
        raise ValueError("Density table does not sum to the source SNP count")
    return rows


def save_data(rows: list[dict[str, int]]) -> None:
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(rows: list[dict[str, int]]) -> None:
    with thesis_style(base_font_size=8.0):
        fig = plt.figure(figsize=(6.3, 6.3), dpi=300)
        ax = fig.add_axes((0.055, 0.17, 0.89, 0.78), projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1.31)
        ax.set_axis_off()

        gap = np.deg2rad(2.4)
        lengths = [max(row["window_end_bp"] for row in rows if row["chromosome"] == c)
                   for c in range(1, 11)]
        usable = 2 * np.pi - 10 * gap
        scale = usable / sum(lengths)
        norm = Normalize(vmin=0, vmax=max(row["snp_count"] for row in rows))
        cmap = plt.get_cmap("viridis")
        start = 0.0

        for chromosome, length in enumerate(lengths, start=1):
            subset = [row for row in rows if row["chromosome"] == chromosome]
            counts = np.array([row["snp_count"] for row in subset])
            end = start + length * scale
            angles = np.linspace(start, end, len(counts) + 1)
            ax.pcolormesh(angles, [0.72, 0.93], counts[np.newaxis, :],
                          cmap=cmap, norm=norm, shading="flat", rasterized=True)
            ax.bar((start + end) / 2, height=0.037, width=end - start,
                   bottom=0.955, color="#28465A", edgecolor="none", align="center")

            for coordinate in range(0, length, 50 * BIN_SIZE):
                angle = start + coordinate * scale
                ax.plot([angle, angle], [0.997, 1.025], color="#536B78", lw=0.6)
                if coordinate and coordinate % (100 * BIN_SIZE) == 0:
                    ax.text(angle, 1.075, str(coordinate // BIN_SIZE),
                            ha="center", va="center", color="#546C78", fontsize=6.7)

            mid = (start + end) / 2
            ax.text(mid, 1.235, f"Chr {chromosome}", ha="center", va="center",
                    color="#183C51", fontsize=8.1, fontweight="bold")
            start = end + gap

        ax.text(0, 0, "LD-pruned SNPs\n2,860,770 variants\n1 Mb windows",
                ha="center", va="center", color="#3C5361", fontsize=8.2,
                linespacing=1.7)
        color_axis = fig.add_axes((0.23, 0.10, 0.54, 0.023))
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                                cax=color_axis, orientation="horizontal")
        colorbar.set_label("SNPs per 1 Mb window", labelpad=5)
        colorbar.outline.set_linewidth(0.5)
        colorbar.ax.tick_params(length=2, labelsize=7)
        fig.text(0.5, 0.032, "Position ticks: 50 Mb; labels: 100 Mb intervals",
                 ha="center", color="#526974", fontsize=7)

        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=300, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    data = load_density()
    save_data(data)
    draw(data)
    print(f"SNPs: {sum(row['snp_count'] for row in data):,}; windows: {len(data):,}")
    print(f"Data: {TABLE}")
    print(f"Figure: {FIGURE}")
