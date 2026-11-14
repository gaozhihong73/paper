"""Plot supplied time-point PCC and RMSE for four pre-specified models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

from dynags_figures.style import thesis_style
from dynags_figures.labels import AXIS_LABELS

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
BASE = ROOT / "05_实验结果" / "5.3" / "P10000_NTR400_H0.8"
DATA = PROJECT / "data"
STEM = "figure_5_8_timepoint_model_performance"
OUTPUT = ROOT / "07_论文图表" / "5-8_代表模型在不同时间点的预测性能.png"
MODELS = {"DynaGS": "dynags", "XGBoost": "xgboost",
          "Random Forest": "random_forest", "Cropformer": "cropformer"}
STYLES = {
    "DynaGS": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "XGBoost": {"color": "#882255", "linestyle": (0, (5, 2)), "marker": "s"},
    "Random Forest": {"color": "#CC79A7", "linestyle": (0, (1, 1.7)), "marker": "^"},
    "Cropformer": {"color": "#AA4499", "linestyle": (0, (5, 1.7, 1, 1.7)), "marker": "D"},
}


def collect() -> pd.DataFrame:
    rows = []
    for model, folder in MODELS.items():
        for repeat in range(1, 11):
            source = BASE / f"rep_{repeat:02d}" / folder / "metrics.json"
            result = json.loads(source.read_text(encoding="utf-8"))
            times = result["by_time"]
            assert len(times) == 20 and [r["time_index"] for r in times] == list(range(20))
            for record in times:
                rows.append({"model": model, "repeat": repeat,
                             "time_index": record["time_index"], "time": record["time"],
                             "pcc": record["pcc"], "rmse": record["rmse"],
                             "valid_n": record["valid_n"], "source": str(source.relative_to(ROOT))})
    data = pd.DataFrame(rows)
    assert len(data) == 800
    assert not data.duplicated(["model", "repeat", "time"]).any()
    assert np.isfinite(data[["pcc", "rmse"]].to_numpy()).all()
    assert (data.valid_n > 0).all()
    return data


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    resamples = np.random.default_rng(73).integers(0, 10, size=(10000, 10))
    rows = []
    for (model, time), group in data.groupby(["model", "time"]):
        group = group.sort_values("repeat")
        assert group.repeat.tolist() == list(range(1, 11))
        for metric in ["pcc", "rmse"]:
            values = group[metric].to_numpy(float)
            low, high = np.quantile(values[resamples].mean(axis=1), [.025, .975])
            rows.append({"model": model, "time": time, "metric": metric, "n_repeats": 10,
                         "mean": values.mean(), "sd": values.std(ddof=1),
                         "ci95_lower": low, "ci95_upper": high})
    return pd.DataFrame(rows)


def draw(summary: pd.DataFrame) -> dict:
    limits = {}
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.05))
        for j, metric in enumerate(["pcc", "rmse"]):
            ax = axes[j]
            # Draw DynaGS last to keep its markers legible where curves overlap.
            for model in ["Cropformer", "Random Forest", "XGBoost", "DynaGS"]:
                part = summary[(summary.model == model) & (summary.metric == metric)].sort_values("time")
                style = STYLES[model]
                ax.fill_between(part.time, part.ci95_lower, part.ci95_upper,
                                color=style["color"], alpha=.10 if model == "DynaGS" else .075,
                                linewidth=0, zorder=1)
                ax.plot(part.time, part["mean"], **style,
                        lw=1.65 if model == "DynaGS" else 1.15,
                        markersize=3.2 if model == "DynaGS" else 2.8,
                        markerfacecolor="white", markeredgewidth=.75,
                        markevery=[0, 3, 6, 9, 12, 15, 19], zorder=4 if model == "DynaGS" else 3)
            part = summary[summary.metric == metric]
            low, high = float(part.ci95_lower.min()), float(part.ci95_upper.max())
            if metric == "pcc":
                ylim = (min(0., low-.04), 1.)
                ax.set_yticks(np.arange(0, 1.01, .2))
            else:
                ylim = (0., high*1.075)
                ax.yaxis.set_major_locator(MultipleLocator(.5))
                ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.1f}"))
            ax.set_ylim(*ylim)
            limits[metric] = list(ylim)
            ax.set_xlim(.6, 20.4)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.tick_params(length=2.5, pad=2, labelsize=7)
            ax.grid(axis="y", color="#DFE7EB", linewidth=.5)
            ax.spines[["left", "bottom"]].set_color("#A7B8C0")
            ax.set_ylabel(AXIS_LABELS[metric],
                          fontsize=8, labelpad=5)
            ax.set_title("A  PCC over time" if metric == "pcc" else "B  RMSE over time",
                         loc="left", fontsize=8.2, color="#193C51", pad=7)
        fig.supxlabel(AXIS_LABELS["time"], y=.175, fontsize=8)
        handles = [Line2D([], [], **STYLES[model], lw=1.65 if model == "DynaGS" else 1.15,
                          markerfacecolor="white", markeredgewidth=.75, markersize=3.2,
                          label=model) for model in MODELS]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .034),
                   ncol=4, fontsize=7.3, columnspacing=1.6, handlelength=2.5, handletextpad=.6)
        fig.subplots_adjust(left=.087, right=.985, top=.88, bottom=.30, wspace=.29)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)
    return limits


if __name__ == "__main__":
    data = collect()
    summary = summarize(data)
    DATA.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA / f"{STEM}_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(DATA / f"{STEM}_summary.csv", index=False, encoding="utf-8-sig")
    limits = draw(summary)
    metadata = {"source_root": str(BASE.relative_to(ROOT)),
                "source_fields": ["metrics.json/by_time/pcc", "metrics.json/by_time/rmse"],
                "source_values": "Supplied 5.3 saved metrics, read directly without prediction-based recomputation.",
                "condition": {"snps": 10000, "n_train": 400, "heritability": .8},
                "models": list(MODELS), "n_repeats": 10, "n_time_points": 20,
                "model_selection": "Pre-specified set used in the experimental figure plan.",
                "line": "Arithmetic mean across 10 independent repetitions at each time point.",
                "band": "Percentile 95% confidence interval of the repeat-level mean, "
                        "10000 bootstrap resamples; seed 73. Identical repeat resamples "
                        "used across models, metrics and times.",
                "markers": "Markers are displayed at selected time points for readability; "
                           "all 20 saved time-point means determine each line and confidence band.",
                "styles": STYLES, "axis_limits": limits, "figure": str(OUTPUT.relative_to(ROOT))}
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Input rows:", len(data), "Model runs:", data[["model", "repeat"]].drop_duplicates().shape[0])
    print(OUTPUT)
