"""Display model ranks within each repeat using supplied 5.3 overall metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch, Rectangle

from dynags_figures.style import thesis_style
from dynags_figures.labels import AXIS_LABELS
from figure_5_6_paired_model_comparison import MODELS, collect

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
STEM = "figure_5_7_repeat_rank_heatmap"
OUTPUT = ROOT / "07_论文图表" / "5-7_不同模型在独立重复中的预测性能排名.png"
METRICS = ["pcc", "r2_time", "rmse"]
ORDER = list(MODELS)


def rank_results(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    for metric in METRICS:
        raw = data.pivot(index="model", columns="repeat", values=metric).loc[ORDER, range(1, 11)]
        # Competition ranking: ties share the best occupied position, e.g. 1,2,2,4.
        ranks = raw.rank(axis=0, ascending=(metric == "rmse"), method="min", na_option="keep")
        assert ranks.loc["DynaGS"].notna().all()
        for model in ORDER:
            values = ranks.loc[model].dropna().to_numpy(float)
            summaries.append({"model": model, "metric": metric, "n_defined_repeats": len(values),
                              "mean_rank": np.mean(values) if len(values) else np.nan,
                              "sd_rank": np.std(values, ddof=1) if len(values) > 1 else np.nan,
                              "best_rank": np.min(values) if len(values) else np.nan,
                              "worst_rank": np.max(values) if len(values) else np.nan})
            for repeat in range(1, 11):
                rows.append({"model": model, "metric": metric, "repeat": repeat,
                             "value": raw.loc[model, repeat], "rank": ranks.loc[model, repeat]})
    long = pd.DataFrame(rows)
    assert len(long) == 360
    assert long[long["rank"].isna()][["model", "metric"]].drop_duplicates().values.tolist() == [["Mean", "pcc"]]
    return long, pd.DataFrame(summaries)


def draw(data: pd.DataFrame) -> None:
    cmap = plt.get_cmap("Blues_r", 12).copy()
    cmap.set_bad("#EDF0F2")
    norm = BoundaryNorm(np.arange(.5, 13, 1), ncolors=12)
    titles = ["A  PCC rank", r"B  $R^2_{\mathrm{time}}$ rank", "C  RMSE rank"]
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        plt.rcParams["hatch.linewidth"] = .35
        fig = plt.figure(figsize=(6.3, 4.25))
        grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, .055],
                               left=.205, right=.936, bottom=.205, top=.895, wspace=.10)
        for j, metric in enumerate(METRICS):
            ax = fig.add_subplot(grid[0, j])
            part = data[data.metric == metric].pivot(index="model", columns="repeat", values="rank")
            values = part.loc[ORDER, range(1, 11)].to_numpy(float)
            im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm,
                           interpolation="nearest", aspect="auto")
            if metric == "pcc":
                ax.add_patch(Rectangle((-.5, -.5), 10, 1, facecolor="#EDF0F2",
                                       edgecolor="#AFBDC6", hatch="///", lw=0, zorder=2))
            ax.set_xticks(np.arange(10), labels=[str(i) for i in range(1, 11)])
            ax.set_yticks(np.arange(len(ORDER)), labels=ORDER if j == 0 else [])
            ax.tick_params(axis="x", length=0, pad=5, labelsize=7)
            ax.tick_params(axis="y", length=0, pad=6, labelsize=7.4)
            ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(ORDER), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.45)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(titles[j], loc="left", fontsize=8, color="#193C51", pad=8)
        color_axis = fig.add_subplot(grid[0, 3])
        colorbar = fig.colorbar(im, cax=color_axis, ticks=[1, 3, 6, 9, 12])
        colorbar.set_label("Rank (1 = best)", fontsize=7.5, labelpad=5)
        colorbar.ax.tick_params(length=2, labelsize=7, pad=2)
        colorbar.outline.set_linewidth(.5)
        colorbar.outline.set_edgecolor("#A7B8C0")
        fig.supxlabel(AXIS_LABELS["repeat"], y=.112, fontsize=8)
        fig.legend(handles=[Patch(facecolor="#EDF0F2", edgecolor="#AFBDC6", hatch="///",
                                  linewidth=.4, label="Undefined PCC")],
                   loc="lower center", bbox_to_anchor=(.5, .023), fontsize=7.3,
                   handlelength=2.0)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    raw = collect()
    ranks, summary = rank_results(raw)
    DATA.mkdir(parents=True, exist_ok=True)
    raw.to_csv(DATA / f"{STEM}_metrics.csv", index=False, encoding="utf-8-sig")
    ranks.to_csv(DATA / f"{STEM}_ranks.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(DATA / f"{STEM}_summary.csv", index=False, encoding="utf-8-sig")
    draw(ranks)
    metadata = {"source": "Direct reading of supplied 5.3 metrics.json overall fields via "
                          "figure_5_6_paired_model_comparison.collect; no prediction-based recomputation.",
                "n_models": 12, "n_repeats": 10,
                "ranking_unit": "Within the same independent repeat, across defined models.",
                "directions": {"pcc": "Descending", "r2_time": "Descending", "rmse": "Ascending"},
                "ties": "Exact ties receive the minimum occupied rank (competition ranking): 1,2,2,4.",
                "undefined": "Mean PCC is excluded from ranking and displayed with a grey hatched row. "
                             "PCC ranks cover 11 models; R2 and RMSE ranks cover 12.",
                "color_limits": [.5, 12.5], "color": "Darker blue denotes a better (smaller) rank.",
                "cell_labels": "Colors only; numeric ranks are retained in the CSV data.",
                "model_order": ORDER, "figure": str(OUTPUT.relative_to(ROOT))}
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary[summary.model == "DynaGS"].to_string(index=False))
    print(OUTPUT)
