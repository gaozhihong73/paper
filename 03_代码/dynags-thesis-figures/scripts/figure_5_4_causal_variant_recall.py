"""Plot exact causal-variant recall of the 20 saved screened SNPs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dynags_figures.style import thesis_style
from dynags_figures.labels import AXIS_LABELS

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
BASE = ROOT / "05_实验结果" / "5.2"
DATA = PROJECT / "data"
STEM = "figure_5_4_causal_variant_recall"
OUTPUT = ROOT / "07_论文图表" / "5-4_不同模拟条件下的因果位点召回率.png"
SIZES = [5000, 10000]
TRAIN = [100, 200, 400, 800, 1600]
H = [.1, .4, .8]
COLORS = {.1: "#0072B2", .4: "#D55E00", .8: "#009E73"}


def collect() -> pd.DataFrame:
    rows = []
    for size in SIZES:
        for n in TRAIN:
            for h in H:
                condition = f"P{size}_NTR{n}_H{h}"
                for repeat in range(1, 11):
                    relative = Path(condition) / f"rep_{repeat:02d}"
                    selection_file = BASE / "experiment2_screened" / relative / "preprocessing/screened_indices.npy"
                    truth_file = BASE / "experiments" / relative / "truth/causal_effects.csv"
                    selected = np.load(selection_file)
                    truth = pd.read_csv(truth_file)["snp_index"].to_numpy(dtype=int) - 1
                    # Both saved NPY indices and raw truth indices are now zero based.
                    oracle = np.load(BASE / "experiment1_oracle" / relative /
                                     "oracle_selection/true_qtl_indices_zero_based.npy")
                    assert selected.ndim == 1 and np.issubdtype(selected.dtype, np.integer)
                    assert len(selected) == len(np.unique(selected)) == 20
                    assert len(truth) == len(np.unique(truth)) == 20
                    assert np.array_equal(np.sort(truth), np.sort(oracle))
                    assert np.all((selected >= 0) & (selected < size))
                    assert np.all((truth >= 0) & (truth < size))
                    recovered = np.intersect1d(selected, truth)
                    rows.append({"snps": size, "n_train": n, "heritability": h,
                                 "repeat": repeat, "selected_count": len(selected),
                                 "true_qtl_count": len(truth), "recovered_count": len(recovered),
                                 "exact_recall_at20": len(recovered) / len(truth),
                                 "selected_indices_zero_based": ";".join(map(str, selected)),
                                 "true_qtl_indices_zero_based": ";".join(map(str, truth)),
                                 "recovered_indices_zero_based": ";".join(map(str, recovered)),
                                 "selection_file": str(selection_file.relative_to(ROOT)),
                                 "truth_file": str(truth_file.relative_to(ROOT))})
    data = pd.DataFrame(rows)
    assert len(data) == 300
    assert (data.groupby(["snps", "n_train", "heritability"]).size() == 10).all()
    return data


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    grouped = data.groupby(["snps", "n_train", "heritability"])["exact_recall_at20"]
    summary = grouped.agg(n_repeats="count", mean="mean", sd="std", median="median",
                          minimum="min", maximum="max")
    summary["q25"] = grouped.quantile(.25)
    summary["q75"] = grouped.quantile(.75)
    return summary.reset_index()


def tied_point_positions(values: np.ndarray, center: int) -> np.ndarray:
    """Spread identical recalls horizontally without altering their actual values."""
    offsets = np.zeros(len(values), dtype=float)
    for level in np.unique(values):
        indices = np.flatnonzero(values == level)
        count = len(indices)
        spacing = min(.09, .72 / max(count-1, 1))
        offsets[indices] = (np.arange(count) - (count-1)/2) * spacing
    return center + offsets


def draw(data: pd.DataFrame) -> None:
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.65), sharex=True, sharey=True)
        for i, size in enumerate(SIZES):
            for j, h in enumerate(H):
                ax = axes[i, j]
                color = COLORS[h]
                subset = data[(data.snps == size) & (data.heritability == h)]
                groups = [subset[subset.n_train == n].sort_values("repeat")["exact_recall_at20"]
                          .to_numpy() for n in TRAIN]
                boxes = ax.boxplot(groups, positions=np.arange(5), widths=.52,
                                   whis=1.5, showfliers=False, patch_artist=True,
                                   manage_ticks=False,
                                   boxprops={"edgecolor": color, "linewidth": .7},
                                   medianprops={"color": "#263F4C", "linewidth": 1.2, "zorder": 6},
                                   whiskerprops={"color": color, "linewidth": .7},
                                   capprops={"color": color, "linewidth": .7})
                for patch in boxes["boxes"]:
                    patch.set_facecolor(matplotlib.colors.to_rgba(color, .15))
                for k, values in enumerate(groups):
                    ax.scatter(tied_point_positions(values, k), values, s=5,
                               color=color, edgecolors="white", linewidths=.25, zorder=4)
                ax.set_title(f"{'ABCDEF'[i*3+j]}  {size:,} SNPs, $h^2={h}$",
                             loc="left", fontsize=7.7, color="#193C51", pad=7)
                ax.set_xlim(-.55, 4.55)
                ax.set_ylim(-.045, 1.045)
                ax.set_xticks(range(5), labels=[str(n) for n in TRAIN])
                ax.set_yticks(np.linspace(0, 1, 6))
                ax.tick_params(length=2.5, pad=2, labelsize=7)
                ax.grid(axis="y", color="#DFE7EB", linewidth=.5)
                ax.spines[["left", "bottom"]].set_color("#A7B8C0")
        fig.supylabel("Exact Recall@20", x=.018, fontsize=8)
        fig.supxlabel(AXIS_LABELS["training_sample_size"], y=.115, fontsize=8)
        fig.legend(handles=[Patch(facecolor="#E5EBEE", edgecolor="#687E8B", linewidth=.7,
                                  label="Median and interquartile range"),
                            Line2D([], [], ls="none", marker="o", color="#455560",
                                   markersize=2.7, label="Individual repeat")],
                   loc="lower center", bbox_to_anchor=(.5, .025), ncol=2,
                   fontsize=7.3, columnspacing=2, handlelength=1.7)
        fig.subplots_adjust(left=.095, right=.98, top=.935, bottom=.205,
                            wspace=.23, hspace=.35)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    data = collect()
    summary = summarize(data)
    DATA.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA / f"{STEM}_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(DATA / f"{STEM}_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {"formula": "Exact Recall@20 = number of true causal SNPs in the saved "
                           "20-SNP selected set / 20 true causal SNPs.",
                "n_conditions": 30, "n_repeats_per_condition": 10, "n_runs": len(data),
                "truth_index_conversion": "Raw causal_effects.csv snp_index is one based; "
                                          "subtract one to match screened_indices.npy. "
                                          "Converted truth matches saved oracle indices in all 300 runs.",
                "boxplot": "Box = 25th-75th percentiles; horizontal line = median; whiskers "
                           "extend to the most extreme actual values within 1.5 IQR. "
                           "Every repeat is shown, including values beyond the whiskers.",
                "points": "One dot per independent repeat. Identical recall values are "
                          "spread horizontally in repeat-number order; no vertical jitter.",
                "figure": str(OUTPUT.relative_to(ROOT))}
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    draw(data)
    print(summary.groupby(["snps", "heritability"])["mean"].agg(["min", "max"]).to_string())
    print(OUTPUT)
