"""Plot the supplied within-model SNP screening gains and 95% CIs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from dynags_figures.style import thesis_style

SOURCE = ROOT / "05_实验结果/5.4/E5_Table.md"
DATA = PROJECT / "data"
STEM = "figure_5_10_screening_benefits"
OUTPUT = ROOT / "07_论文图表/5-10_同一模型筛选前后的预测收益.png"
MODELS = ["Ridge-All", "RR-GBLUP", "Random Forest", "XGBoost", "DNNGP-SNP",
          "Cropformer", "DynaGS"]
SOURCE_LABELS = {"Ridge": "Ridge-All"}


def collect() -> pd.DataFrame:
    """Read reported summaries without reconstructing individual repeats."""
    content = SOURCE.read_text(encoding="utf-8-sig")
    section = content.split("## Table E5a.", 1)[1].split("## Table E5b.", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        model = SOURCE_LABELS.get(cells[0], cells[0])
        if model not in MODELS:
            continue
        if len(cells) != 4:
            raise ValueError(f"Unexpected summary row: {line}")
        for metric, cell in zip(["pcc", "r2", "rmse"], cells[1:], strict=True):
            match = re.fullmatch(r"([\d.+-]+)\s*\[([\d.+-]+),\s*([\d.+-]+)\]", cell)
            if match is None:
                raise ValueError(f"Unexpected interval: {cell}")
            mean, lower, upper = map(float, match.groups())
            if not np.isfinite([mean, lower, upper]).all() or not lower <= mean <= upper:
                raise ValueError(f"Invalid reported interval: {cell}")
            rows.append({"model": model, "source_model": cells[0], "metric": metric, "mean_difference": mean,
                         "ci95_lower": lower, "ci95_upper": upper, "n_repeats": 10})
    data = pd.DataFrame(rows)
    if len(data) != 21 or data.duplicated(["model", "metric"]).any():
        raise ValueError("Expected three summaries for each of seven models")
    if list(data.model.drop_duplicates()) != MODELS:
        raise ValueError("Model order differs from the supplied table")
    return data


def draw(data: pd.DataFrame) -> dict:
    specs = [
        ("pcc", "A  PCC improvement", r"$\Delta\mathrm{PCC}$", (-.020, .400), .1, "%.1f"),
        ("rmse", "B  RMSE reduction", r"$\Delta\mathrm{RMSE}$", (-.008, .160), .04, "%.2f"),
    ]
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.35), sharey=True)
        for ax, (metric, title, xlabel, limits, tick, fmt) in zip(axes, specs, strict=True):
            part = data[data.metric == metric].set_index("model").loc[MODELS]
            for i, row in enumerate(part.itertuples()):
                if i % 2 == 0:
                    ax.axhspan(i - .5, i + .5, color="#F2F6F8", linewidth=0, zorder=0)
                ax.errorbar(row.mean_difference, i,
                            xerr=np.array([[row.mean_difference - row.ci95_lower],
                                           [row.ci95_upper - row.mean_difference]]),
                            fmt="o", color="#0072B2", markersize=4.6,
                            markeredgecolor="white", markeredgewidth=.55,
                            elinewidth=1.15, capsize=2.8, capthick=.9, zorder=3)
            ax.axvline(0, color="#8195A0", linestyle=(0, (3, 2)), linewidth=.85, zorder=2)
            ax.set_xlim(*limits)
            ax.set_ylim(len(MODELS) - .5, -.5)
            ax.set_yticks(np.arange(len(MODELS)), labels=MODELS)
            ax.xaxis.set_major_locator(MultipleLocator(tick))
            ax.xaxis.set_major_formatter(FormatStrFormatter(fmt))
            ax.tick_params(axis="y", length=0, pad=6, labelsize=8)
            ax.tick_params(axis="x", length=2.5, pad=3, labelsize=7.3)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#A7B8C0")
            ax.grid(axis="x", color="#E2E9ED", linewidth=.5)
            ax.set_title(title, loc="left", fontsize=8.5, color="#193C51", pad=10)
            ax.set_xlabel(xlabel, fontsize=8.4, labelpad=6)
        axes[1].tick_params(axis="y", labelleft=False)
        fig.text(.5, .055, "Positive values indicate a screening benefit",
                 ha="center", va="center", fontsize=7.7, color="#455560")
        fig.subplots_adjust(left=.185, right=.98, top=.89, bottom=.215, wspace=.20)

        # Check text against the final, fixed-size publication canvas.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.bbox
        for item in fig.findobj(matplotlib.text.Text):
            if not item.get_visible() or not item.get_text():
                continue
            box = item.get_window_extent(renderer)
            if (box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1
                    or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1):
                # Locator ticks outside the axis range are not rendered.
                if item in [label for axis in axes for label in axis.get_xticklabels()]:
                    continue
                raise ValueError(f"Text extends outside the canvas: {item.get_text()}")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)
    return {metric: list(limits) for metric, _, _, limits, _, _ in specs}


def main() -> None:
    data = collect()
    DATA.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA / f"{STEM}_summary.csv", index=False, encoding="utf-8-sig")
    limits = draw(data)
    metadata = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_section": "Table E5a. Effect of SNP screening",
        "source_values": "Reported means and 95% confidence limits read directly from the supplied table.",
        "n_repeats": 10,
        "point": "Mean within-model paired screening benefit reported in the source.",
        "interval": "95% confidence interval reported in the source; used without recalculation.",
        "directions": {"pcc": "PCC_screened minus PCC_all",
                       "r2": "R2_screened minus R2_all",
                       "rmse": "RMSE_all minus RMSE_screened"},
        "models": MODELS,
        "ridge_label": {
            "source_table_label": "Ridge",
            "display_label": "Ridge-All",
            "encoding": "Additive SNP dosage only; no dominance features.",
            "comparison": "The same additive Ridge model with all SNPs versus screened SNPs.",
            "run_record": "05_实验结果/5.4/P10000_NTR400_H0.8/rep_01/ridge_all/run_manifest.json",
            "registry": "https://github.com/gaozhihong73/Code/blob/main/experiments/e3/baselines/registry.py#L40-L43",
            "implementation": "https://github.com/gaozhihong73/Code/blob/main/experiments/e3/baselines/adapters.py#L89-L104",
        },
        "displayed_metrics": ["pcc", "rmse"],
        "axis_limits": limits,
        "figure_size_inches": [6.3, 3.35],
        "dpi": 600,
        "figure": str(OUTPUT.relative_to(ROOT)),
        "caption": "图 5-10 同一模型筛选前后的预测收益",
        "caption_note": "A：PCC 提升；B：RMSE 降低。点表示 10 次独立重复的配对收益均值，误差线表示所提供汇总结果中的 95% 置信区间。ΔPCC 为筛选后减筛选前，ΔRMSE 为筛选前减筛选后，正值均表示筛选带来预测收益。Ridge-All 表示仅采用加性剂量编码的岭回归，该行比较完整 SNP 与筛选 SNP 两种输入。",
    }
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    print(data.to_string(index=False))
    print(OUTPUT)


if __name__ == "__main__":
    main()
