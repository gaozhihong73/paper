"""Heatmaps of repeat-paired simulation differences; positive means screening benefit."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
SOURCE = DATA / "figure_5_2_simulation_performance_runs.csv"
STEM = "figure_5_3_paired_performance_differences"
OUTPUT = ROOT / "07_论文图表" / "5-3_两种SNP输入设置的配对预测性能差异.png"
METRICS = ["pcc", "r2_time", "rmse"]
SIZES = [5000, 10000]
TRAIN = [100, 200, 400, 800, 1600]
H = [.8, .4, .1]


def prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE)
    keys = ["snps", "n_train", "heritability", "repeat"]
    assert not source.duplicated(keys + ["setting"]).any()
    oracle = source[source.setting == "Oracle"].set_index(keys)
    screened = source[source.setting == "Screened"].set_index(keys)
    assert len(oracle) == len(screened) == 300
    paired = oracle[[]].copy()
    for metric in METRICS:
        paired[metric] = (screened[metric] - oracle[metric]) * (-1 if metric == "rmse" else 1)
    assert np.isfinite(paired[METRICS].to_numpy()).all()
    paired = paired.reset_index()
    rows = []
    for values, group in paired.groupby(keys[:-1]):
        assert len(group) == 10 and set(group.repeat) == set(range(1, 11))
        for metric in METRICS:
            rows.append(dict(zip(keys[:-1], values)) |
                        {"metric": metric, "mean_difference": group[metric].mean(),
                         "sd_difference": group[metric].std(ddof=1), "n_pairs": len(group)})
    return paired, pd.DataFrame(rows)


def draw(summary: pd.DataFrame) -> dict:
    cmap = LinearSegmentedColormap.from_list("screening_benefit", ["#B5502A", "#FAFAF7", "#0072B2"])
    labels = [r"$\Delta\mathrm{PCC}$", r"$\Delta R^2_{\mathrm{time}}$", r"$\Delta\mathrm{RMSE}$"]
    limits = {}
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig = plt.figure(figsize=(6.3, 5.05))
        # The colorbar axes belong to the same GridSpec row as their heatmaps.
        grid = fig.add_gridspec(3, 3, width_ratios=[1, 1, .045],
                               left=.105, right=.895, bottom=.16, top=.94,
                               wspace=.20, hspace=.53)
        for row, metric in enumerate(METRICS):
            step = .05 if metric == "pcc" else .02
            bound = float(np.ceil(summary.loc[summary.metric == metric, "mean_difference"].abs().max()
                                  / step) * step)
            limits[metric] = [-bound, bound]
            for col, size in enumerate(SIZES):
                ax = fig.add_subplot(grid[row, col])
                subset = summary[(summary.metric == metric) & (summary.snps == size)]
                values = subset.pivot(index="heritability", columns="n_train",
                                      values="mean_difference").loc[H, TRAIN].to_numpy()
                im = ax.imshow(values, cmap=cmap, vmin=-bound, vmax=bound, aspect="auto",
                               interpolation="nearest")
                for i in range(3):
                    for j in range(5):
                        v = values[i, j]
                        # Display zero without a sign when rounding removes the magnitude.
                        number = "0.000" if abs(v) < .0005 else f"{v:+.3f}".replace("-", "−")
                        ax.text(j, i, number, ha="center", va="center", fontsize=7.2,
                                color="white" if abs(v) > .58*bound else "#253C49")
                ax.set_xticks(range(5), labels=[str(n) for n in TRAIN])
                ax.set_yticks(range(3), labels=[str(h) for h in H])
                if col == 0:
                    ax.set_ylabel(r"Heritability ($h^2$)", labelpad=5)
                ax.set_title(f"{'ABCDEF'[row*2+col]}  {size:,} SNPs", loc="left",
                             fontsize=8, color="#193C51", pad=6)
                ax.tick_params(length=0, pad=4, labelsize=7)
                ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
                ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
                ax.grid(which="minor", color="white", linewidth=.7)
                ax.tick_params(which="minor", length=0)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            bar_ax = fig.add_subplot(grid[row, 2])
            colorbar = fig.colorbar(im, cax=bar_ax, ticks=np.linspace(-bound, bound, 5))
            colorbar.set_label(labels[row], labelpad=5, fontsize=8)
            colorbar.ax.tick_params(labelsize=6.8, length=2, pad=2)
            colorbar.outline.set_linewidth(.45)
            colorbar.outline.set_edgecolor("#A7B8C0")
        fig.supxlabel("Training individuals", y=.087, fontsize=8)
        fig.text(.5, .033, "Positive values indicate a screening benefit", ha="center",
                 va="center", fontsize=7.5, color="#455560")
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)
    return limits


if __name__ == "__main__":
    pairs, cells = prepare()
    pairs.to_csv(DATA / f"{STEM}_pairs.csv", index=False, encoding="utf-8-sig")
    cells.to_csv(DATA / f"{STEM}_summary.csv", index=False, encoding="utf-8-sig")
    limits = draw(cells)
    metadata = {"source": str(SOURCE.relative_to(ROOT)), "n_pairs_per_cell": 10,
                "aggregation": "Within each condition, calculate the paired difference for each "
                               "repeat, then the arithmetic mean over 10 repeats.",
                "directions": {"pcc": "Screened minus Oracle", "r2_time": "Screened minus Oracle",
                               "rmse": "Oracle minus Screened"},
                "color": "Orange = negative; white = zero; blue = positive. "
                         "Symmetric limits shared by SNP sizes within each metric.",
                "color_limits": limits,
                "annotation": "Mean paired difference rounded to 3 decimal places; no significance tests.",
                "figure": str(OUTPUT.relative_to(ROOT))}
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(cells.groupby("metric")["mean_difference"].agg(["min", "max"]).to_string())
    print(OUTPUT)
