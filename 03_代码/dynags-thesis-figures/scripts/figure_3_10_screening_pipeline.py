"""Draw the train-only, two-stage SNP screening pipeline in Section 3.5.

Method source: gaozhihong73/Code, utils/Screening.py, screen_snps.
This is a method diagram; its marker counts are symbolic, not run results.
"""

from __future__ import annotations

from itertools import pairwise
from math import hypot
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "07_论文图表" / "3-10_动态关联分析与距离相关性的SNP分阶段筛选流程.png"

INK = "#193B50"
MUTED = "#506A7A"
LINE = "#708A98"
BLUE_FACE = "#EAF3F8"
BLUE_EDGE = "#6A9AB5"
AMBER_FACE = "#FCF3E8"
AMBER_EDGE = "#C6A06F"
GREEN_FACE = "#E8F4EF"
GREEN_EDGE = "#70A78B"
GRAY_FACE = "#F2F5F6"
GRAY_EDGE = "#A6B5BC"


def card(ax, x, y, w, h, *, title, lines, face, edge, title_size=8.6):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.055",
            linewidth=1.0, edgecolor=edge, facecolor=face, zorder=2,
        )
    )
    center = x + w / 2
    ax.text(center, y + h * 0.70, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=INK, zorder=3)
    if len(lines) == 1:
        ys = [y + h * 0.30]
    elif len(lines) == 2:
        ys = [y + h * 0.40, y + h * 0.17]
    else:
        ys = [y + h * 0.48, y + h * 0.29, y + h * 0.11]
    for line, yy in zip(lines, ys, strict=True):
        ax.text(center, yy, line, ha="center", va="center",
                fontsize=7.6, color=MUTED, zorder=3)


def arrow(ax, start, end, *, color=LINE, lw=1.25, end_gap=0.035):
    # Stop the arrowhead just before the destination card's border.
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = hypot(dx, dy)
    tip_gap = min(end_gap, length / 4)
    tip = (end[0] - dx / length * tip_gap,
           end[1] - dy / length * tip_gap)
    ax.add_patch(
        FancyArrowPatch(start, tip, arrowstyle="-|>", mutation_scale=10,
                        linewidth=lw, color=color, shrinkA=0, shrinkB=0,
                        connectionstyle="arc3,rad=0", zorder=1)
    )


def elbow(ax, points, *, color=LINE, end_gap=0.035):
    for first, second in pairwise(points[:-1]):
        ax.plot([first[0], second[0]], [first[1], second[1]],
                color=color, linewidth=1.15, solid_capstyle="round", zorder=1)
    arrow(ax, points[-2], points[-1], color=color, end_gap=end_gap)


def draw():
    with thesis_style(base_font_size=8.0):
        fig, ax = plt.subplots(figsize=(6.3, 5.2), dpi=300)
        ax.set_xlim(0, 6.3)
        ax.set_ylim(0, 5.2)
        ax.axis("off")

        left_x, left_w = 0.33, 2.14
        right_x, right_w = 3.50, 2.14
        left_center = left_x + left_w / 2
        right_center = right_x + right_w / 2

        card(ax, left_x, 4.39, left_w, 0.66,
             title="Training SNP dosage",
             lines=[r"$N_{\mathrm{tr}} \times M_0$; original marker indices"],
             face=BLUE_FACE, edge=BLUE_EDGE)
        card(ax, right_x, 4.39, right_w, 0.66,
             title="Training phenotypes",
             lines=[r"$N_{\mathrm{tr}} \times P \times T$; time coordinates"],
             face=AMBER_FACE, edge=AMBER_EDGE)

        card(ax, left_x, 3.40, left_w, 0.76,
             title="SNP quality control & filling",
             lines=["Raw dosage: MAC ≥ 5; missing ≤ 20%",
                    r"Train-fitted mode fill; $M_{\mathrm{QC}}$ SNPs"],
             face=BLUE_FACE, edge=BLUE_EDGE)
        card(ax, right_x, 3.40, right_w, 0.76,
             title="Dynamic trait extraction",
             lines=["Mean · AUC · slope", "Rank-normalized traits"],
             face=AMBER_FACE, edge=AMBER_EDGE, title_size=8.0)

        card(ax, left_x, 2.39, left_w, 0.79,
             title="Additive–dominance joint GWAS",
             lines=["PC-adjusted, 2-df test",
                    r"Minimum $P$ across traits → top $K_{\mathrm{GWAS}}$"],
             face=BLUE_FACE, edge=BLUE_EDGE, title_size=8.3)
        card(ax, right_x, 2.39, right_w, 0.79,
             title="Full phenotype vector",
             lines=[r"$P \times T$ values per individual",
                    "For distance correlation"],
             face=AMBER_FACE, edge=AMBER_EDGE, title_size=8.1)

        card(ax, left_x, 1.38, left_w, 0.79,
             title="DC-SIS refinement",
             lines=["SNP dosage vs. full trajectory",
                    r"Distance correlation → top $K_{\mathrm{final}}$"],
             face=GREEN_FACE, edge=GREEN_EDGE)

        card(ax, left_x, 0.32, left_w, 0.81,
             title="Final SNP indices",
             lines=["Restored to original column order",
                    "Applied to train / validation / test"],
             face=GRAY_FACE, edge=GRAY_EDGE)

        # Vertical genotype branch.
        for top, bottom in [(4.39, 4.16), (3.40, 3.18),
                            (2.39, 2.17), (1.38, 1.13)]:
            arrow(ax, (left_center, top), (left_center, bottom))

        # The phenotype summaries feed GWAS; the complete trajectory feeds DC-SIS.
        arrow(ax, (right_center, 4.39), (right_center, 4.16), color=AMBER_EDGE)
        elbow(ax, [(right_x, 3.78), (3.17, 3.78),
                   (3.17, 2.79), (left_x + left_w, 2.79)], color=AMBER_EDGE)
        elbow(ax, [(right_x + right_w, 4.72), (5.99, 4.72),
                   (5.99, 2.79), (right_x + right_w, 2.79)],
              color=AMBER_EDGE, end_gap=0.0)
        elbow(ax, [(right_x, 2.79), (3.27, 2.79),
                   (3.27, 1.77), (left_x + left_w, 1.77)], color=AMBER_EDGE)

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=300, facecolor="white")
        plt.close(fig)
        return OUTPUT


if __name__ == "__main__":
    print(draw())
