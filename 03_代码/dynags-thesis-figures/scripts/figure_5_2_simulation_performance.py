"""Plot simulation performance over all 30 conditions and 10 repeats."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import t as student_t

from dynags_figures.style import thesis_style
from dynags_figures.labels import AXIS_LABELS

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
BASE = ROOT / "05_实验结果" / "5.2"
OUTPUT = ROOT / "07_论文图表" / "5-2_不同模拟条件下的纵向表型预测性能.png"
STEM = "figure_5_2_simulation_performance"
SIZES = [5000, 10000]
TRAIN = [100, 200, 400, 800, 1600]
HERITABILITY = [.1, .4, .8]
SETTINGS = {"Oracle": "experiment1_oracle", "Screened": "experiment2_screened"}
METRICS = ["pcc", "r2_time", "rmse"]


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    """Original-scale metrics following the equations in thesis section 5.1.4."""
    y = y.astype(np.float64)
    prediction = prediction.astype(np.float64)
    assert y.shape == prediction.shape and y.ndim == 2
    assert np.isfinite(y).all() and np.isfinite(prediction).all()
    error = y - prediction
    a = y - y.mean(axis=0)
    b = prediction - prediction.mean(axis=0)
    denominator = np.sqrt((a*a).sum(axis=0) * (b*b).sum(axis=0))
    valid = denominator >= 1e-10
    pcc = float(np.mean((a*b).sum(axis=0)[valid] / denominator[valid])) if valid.any() else np.nan
    total = float((a*a).sum())
    return {"pcc": pcc, "r2_time": 1-float((error*error).sum())/total if total >= 1e-10 else 0.,
            "rmse": float(np.sqrt(np.mean(error*error, axis=0)).mean()),
            "pcc_valid_timepoints": int(valid.sum()), "total_timepoints": y.shape[1]}


def collect() -> pd.DataFrame:
    rows = []
    for size in SIZES:
        for n in TRAIN:
            for h in HERITABILITY:
                condition = f"P{size}_NTR{n}_H{h}"
                for repeat in range(1, 11):
                    reference = None
                    for setting, folder in SETTINGS.items():
                        source = BASE / folder / condition / f"rep_{repeat:02d}" / "test_predictions.npz"
                        with np.load(source) as archive:
                            y = archive["targets"][:, 0, :]
                            predicted = archive["predictions"][:, 0, :]
                            indices = archive["sample_idx"]
                            if reference is None:
                                reference = (indices.copy(), y.copy())
                            else:
                                assert np.array_equal(indices, reference[0])
                                assert np.allclose(y, reference[1])
                            values = metrics(y, predicted)
                        rows.append({"snps": size, "n_train": n, "heritability": h,
                                     "setting": setting, "repeat": repeat,
                                     "n_test": len(y), **values,
                                     "prediction_file": str(source.relative_to(ROOT))})
    data = pd.DataFrame(rows)
    assert len(data) == 600 and data[METRICS].notna().all().all()
    return data


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, part in data.groupby(["snps", "n_train", "heritability", "setting"]):
        for metric in METRICS:
            values = part[metric].to_numpy(float)
            n = len(values)
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            half = float(student_t.ppf(.975, n-1)*sd/np.sqrt(n))
            rows.append(dict(zip(["snps", "n_train", "heritability", "setting"], group)) |
                        {"metric": metric, "n_repeats": n, "mean": mean, "sd": sd,
                         "ci95_lower": mean-half, "ci95_upper": mean+half})
    return pd.DataFrame(rows)


def draw(summary: pd.DataFrame) -> None:
    colors = {.1: "#0072B2", .4: "#D55E00", .8: "#009E73"}
    labels = ["PCC", r"$R^2_{\mathrm{time}}$", "RMSE"]
    styles = {"Oracle": {"linestyle": (0, (4, 2)), "marker": "o"},
              "Screened": {"linestyle": "-", "marker": "s"}}
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.8), sharex=True, sharey="col")
        for i, size in enumerate(SIZES):
            for j, metric in enumerate(METRICS):
                ax = axes[i, j]
                for h in HERITABILITY:
                    for setting in SETTINGS:
                        part = summary[(summary.snps == size) & (summary.metric == metric) &
                                       (summary.heritability == h) & (summary.setting == setting)]
                        part = part.sort_values("n_train")
                        x = part["n_train"].to_numpy()
                        ax.fill_between(x, part.ci95_lower, part.ci95_upper,
                                        color=colors[h], alpha=.065, linewidth=0, zorder=1)
                        ax.plot(x, part["mean"], color=colors[h], linewidth=1.05,
                                markersize=3, markeredgewidth=.7,
                                markerfacecolor="white" if setting == "Oracle" else colors[h],
                                **styles[setting], zorder=3)
                ax.set_xscale("log", base=2)
                ax.set_xlim(85, 1880)
                ax.set_xticks(TRAIN, labels=[str(n) for n in TRAIN])
                ax.minorticks_off()
                ax.set_title(f"{'ABCDEF'[i*3+j]}  {size:,} SNPs", loc="left",
                             fontsize=8, pad=6, color="#193C51")
                ax.set_ylabel(labels[j], fontsize=8, labelpad=3)
                ax.tick_params(axis="both", labelsize=6.8, length=2.5, pad=2)
                ax.grid(axis="y", color="#DFE7EB", linewidth=.5)
                ax.spines[["left", "bottom"]].set_color("#A7B8C0")
                if metric == "r2_time":
                    ax.axhline(0, color="#8CA0AA", linewidth=.65, ls=":", zorder=0)
        legend_h = [Line2D([], [], color=colors[h], lw=1.6, label=rf"$h^2={h}$")
                    for h in HERITABILITY]
        fig.legend(handles=legend_h, loc="lower center", bbox_to_anchor=(.5, .087),
                   ncol=3, fontsize=7.5, columnspacing=2.8, handlelength=2.1)
        legend_s = [Line2D([], [], color="#455560", markerfacecolor="white" if s == "Oracle"
                          else "#455560", lw=1.05, markersize=3, **styles[s], label=s+" SNPs")
                    for s in SETTINGS]
        fig.legend(handles=legend_s, loc="lower center", bbox_to_anchor=(.5, .033),
                   ncol=2, fontsize=7.5, columnspacing=3.4, handlelength=2.8)
        fig.supxlabel(AXIS_LABELS["training_sample_size"], y=.165, fontsize=8)
        fig.subplots_adjust(left=.078, right=.985, top=.94, bottom=.25,
                            wspace=.36, hspace=.34)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    data = collect()
    summary = summarize(data)
    destination = PROJECT / "data"
    data.to_csv(destination / f"{STEM}_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(destination / f"{STEM}_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {"runs": len(data), "conditions": 30, "repeats_per_setting_per_condition": 10,
                "interval": "Two-sided Student t 95% confidence interval of the repeat mean, "
                            "mean +/- t(0.975,9) * sample SD / sqrt(10).",
                "metrics": {"pcc": "Mean across defined time-point Pearson correlations over individuals.",
                            "r2_time": "1 - total SSE / sum of squared deviations from each test time mean.",
                            "rmse": "Arithmetic mean of original-scale time-point RMSE values."},
                "x_axis": AXIS_LABELS["training_sample_size"] + ", logarithmic scale base 2.",
                "figure": str(OUTPUT.relative_to(ROOT))}
    (destination / f"{STEM}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    draw(summary)
    print(summary.groupby("metric")["mean"].agg(["min", "max"]).to_string())
    print(OUTPUT)
