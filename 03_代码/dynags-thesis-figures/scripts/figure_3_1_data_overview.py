"""Draw the maize genotype and longitudinal phenotype data assembly flow."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from figure_3_4_trait_trajectories import TIMES, TRAITS
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
DATA = ROOT / "04_数据" / "真实玉米数据集"
SELECTION = PROJECT / "data" / "figure_3_8_representative_selection.csv"
SUMMARY = PROJECT / "data" / "figure_3_1_data_assembly_summary.csv"
FIGURE = ROOT / "07_论文图表" / "3-1_真实玉米数据组成与匹配流程.png"


def inspect_inputs() -> dict[str, int]:
    samples = pd.read_csv(DATA / "plink" / "data_ld_pruned.psam", sep="\t", dtype=str)
    if "#IID" not in samples.columns or samples["#IID"].isna().any():
        raise ValueError("Missing genotype individual IDs")
    if samples["#IID"].duplicated().any():
        raise ValueError("Duplicate genotype individual IDs")
    n_individuals = len(samples)
    if n_individuals != 400:
        raise ValueError(f"Unexpected genotype sample count: {n_individuals}")

    log = (DATA / "plink" / "data_ld_pruned.log").read_text(encoding="utf-8")
    match = re.search(r"(\d+) variants remaining after main filters", log)
    if match is None:
        raise ValueError("No LD-pruned SNP count in PLINK log")
    n_snps = int(match.group(1))
    if n_snps != 2_860_770:
        raise ValueError(f"Unexpected LD-pruned SNP count: {n_snps}")

    files = {path.stem for path in (DATA / "pheno").glob("*.csv")}
    if files != set(TRAITS):
        raise ValueError("Phenotype files do not match the 24 candidate traits")
    for trait in TRAITS:
        frame = pd.read_csv(DATA / "pheno" / f"{trait}.csv")
        if frame.shape != (n_individuals, len(TIMES)):
            raise ValueError(f"Unexpected phenotype dimensions: {trait}")
        if list(frame.columns) != list(map(str, TIMES)):
            raise ValueError(f"Unexpected phenotype time points: {trait}")

    chosen = pd.read_csv(SELECTION)
    if len(chosen) != 8 or chosen.selected_trait.nunique() != 8:
        raise ValueError("Expected eight unique representative traits")
    if not set(chosen.selected_trait).issubset(files):
        raise ValueError("Selected trait absent from the phenotype data")
    return {
        "individuals": n_individuals,
        "ld_pruned_candidate_snps": n_snps,
        "candidate_traits": len(TRAITS),
        "selected_traits": len(chosen),
        "time_points": len(TIMES),
    }


def save_data(counts: dict[str, int]) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("Genotype", "individuals", counts["individuals"], "plink/data_ld_pruned.psam"),
        ("Genotype", "LD-pruned candidate SNPs", counts["ld_pruned_candidate_snps"],
         "plink/data_ld_pruned.log"),
        ("Phenotype", "individuals per trait", counts["individuals"], "pheno/*.csv"),
        ("Phenotype", "candidate traits", counts["candidate_traits"], "pheno/*.csv"),
        ("Phenotype", "time points", counts["time_points"], "pheno/*.csv"),
        ("Selected set", "representative traits", counts["selected_traits"],
         "figure_3_8_representative_selection.csv"),
    ]
    with SUMMARY.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "measure", "value", "source"])
        writer.writerows(rows)


def draw(counts: dict[str, int]) -> None:
    def card(ax, xy, size, *, face: str, edge: str, title: str,
             detail: str, extra: str | None = None) -> None:
        x, y = xy
        width, height = size
        ax.add_patch(
            FancyBboxPatch(
                xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=face, edgecolor=edge, linewidth=1.0,
            )
        )
        center = x + width / 2
        ax.text(center, y + height * 0.73, title, ha="center", va="center",
                color="#193B50", fontsize=8.5, fontweight="bold")
        ax.text(center, y + height * 0.43, detail, ha="center", va="center",
                color="#274C60", fontsize=8.6)
        if extra:
            ax.text(center, y + height * 0.18, extra, ha="center", va="center",
                    color="#5A7080", fontsize=7.0)

    with thesis_style(base_font_size=8.0):
        fig, ax = plt.subplots(figsize=(6.3, 4.6), dpi=300)
        ax.set_xlim(0, 6.3)
        ax.set_ylim(0, 4.6)
        ax.axis("off")

        card(
            ax, (0.20, 3.24), (2.72, 0.98), face="#EAF3F8", edge="#6A9AB5",
            title="LD-pruned genotype data",
            detail=f"{counts['individuals']} individuals × {counts['ld_pruned_candidate_snps']:,} SNPs",
            extra="Candidate marker set",
        )
        card(
            ax, (3.38, 3.24), (2.72, 0.98), face="#FCF3E8", edge="#C6A06F",
            title="Longitudinal phenotype data",
            detail=f"{counts['individuals']} individuals × {counts['candidate_traits']} traits",
            extra=f"{counts['time_points']} observed time points per trait",
        )
        card(
            ax, (1.70, 1.76), (2.90, 0.88), face="#F4F7F8", edge="#8096A3",
            title="Matched maize population",
            detail=f"{counts['individuals']} individuals",
            extra="Matched by individual ID",
        )
        card(
            ax, (1.70, 0.28), (2.90, 0.88), face="#EAF3F8", edge="#176B91",
            title="Selected trait analysis set",
            detail=(f"{counts['individuals']} individuals × "
                    f"{counts['selected_traits']} traits × {counts['time_points']} time points"),
            extra="One representative trait per trajectory cluster",
        )

        arrow_style = {
            "arrowstyle": "-|>",
            "mutation_scale": 11,
            "linewidth": 1.1,
            "color": "#5B7788",
            "connectionstyle": "arc3,rad=0",
        }
        ax.add_patch(FancyArrowPatch((1.56, 3.19), (2.48, 2.69), **arrow_style))
        ax.add_patch(FancyArrowPatch((4.74, 3.19), (3.82, 2.69), **arrow_style))
        ax.add_patch(FancyArrowPatch((3.15, 1.70), (3.15, 1.22), **arrow_style))

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.14)
        plt.close(fig)


if __name__ == "__main__":
    values = inspect_inputs()
    save_data(values)
    draw(values)
    print(f"Counts: {values}")
    print(f"Data: {SUMMARY}")
    print(f"Figure: {FIGURE}")
