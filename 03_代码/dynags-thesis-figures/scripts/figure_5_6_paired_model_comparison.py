"""Forest plots using the supplied 5.3 per-run overall metrics directly."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
BASE = ROOT / "05_实验结果" / "5.3" / "P10000_NTR400_H0.8"
DATA = PROJECT / "data"
STEM = "figure_5_6_paired_model_comparison"
OUTPUT = ROOT / "07_论文图表" / "5-6_DynaGS与对比模型的配对预测性能差异.png"
MODELS = {
    "Mean": "mean", "Ridge-All": "ridge_all", "Ridge-A+D": "ridge_a_plus_d",
    "GBLUP-All": "gblup_all", "RR-GBLUP": "rr_gblup",
    "Random Forest": "random_forest", "XGBoost": "xgboost",
    "DNNGP-SNP": "dnngp_snp", "DLGWAS": "dlgwas", "WheatGP": "wheatgp",
    "Cropformer": "cropformer", "DynaGS": "dynags",
}
BASELINES = [model for model in MODELS if model != "DynaGS"]


def collect() -> pd.DataFrame:
    rows = []
    for repeat in range(1, 11):
        for model, folder in MODELS.items():
            source = BASE / f"rep_{repeat:02d}" / folder / "metrics.json"
            result = json.loads(source.read_text(encoding="utf-8"))
            values = result["overall"]
            rows.append({"repeat": repeat, "model": model,
                         "pcc": np.nan if values["pcc"] is None else float(values["pcc"]),
                         "r2_time": float(values["time_adjusted_r2"]),
                         "rmse": float(values["rmse"]),
                         "source": str(source.relative_to(ROOT))})
    data = pd.DataFrame(rows)
    assert len(data) == 120 and not data.duplicated(["repeat", "model"]).any()
    assert data[data.model == "Mean"].pcc.isna().all()
    assert np.isfinite(data[data.model != "Mean"].pcc).all()
    assert np.isfinite(data.rmse).all()
    return data


def compare(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = data[data.model == "DynaGS"].set_index("repeat").sort_index()
    pair_rows = []
    summary_rows = []
    resamples = np.random.default_rng(73).integers(0, 10, size=(10000, 10))
    for model in BASELINES:
        baseline = data[data.model == model].set_index("repeat").sort_index()
        assert baseline.index.equals(reference.index)
        for metric in ["pcc", "rmse"]:
            delta = (reference[metric] - baseline[metric]) * (-1 if metric == "rmse" else 1)
            for repeat, value in delta.items():
                pair_rows.append({"baseline": model, "metric": metric, "repeat": repeat,
                                  "dynags_value": reference.loc[repeat, metric],
                                  "baseline_value": baseline.loc[repeat, metric],
                                  "paired_difference": value})
            valid = delta.notna()
            if not valid.any():
                assert model == "Mean" and metric == "pcc"
                mean = sd = low = high = np.nan
            else:
                assert valid.all() and len(delta) == 10
                values = delta.to_numpy(float)
                mean, sd = float(values.mean()), float(values.std(ddof=1))
                low, high = np.quantile(values[resamples].mean(axis=1), [.025, .975])
            summary_rows.append({"baseline": model, "metric": metric, "n_pairs": int(valid.sum()),
                                 "mean_difference": mean, "sd_difference": sd,
                                 "ci95_lower": low, "ci95_upper": high})
    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


def draw(summary: pd.DataFrame) -> dict:
    limits = {}
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.25), sharey=True)
        y = np.arange(len(BASELINES))
        for j, metric in enumerate(["pcc", "rmse"]):
            ax = axes[j]
            part = summary[summary.metric == metric].set_index("baseline").loc[BASELINES]
            finite = part[part.n_pairs > 0]
            lower = min(0., float(finite.ci95_lower.min()))
            upper = max(0., float(finite.ci95_upper.max()))
            span = upper-lower
            xlim = (lower-.065*span, upper+.10*span)
            limits[metric] = list(xlim)
            for index in y:
                if index % 2 == 0:
                    ax.axhspan(index-.5, index+.5, color="#F2F6F8", lw=0, zorder=0)
                row = part.iloc[index]
                if row.n_pairs == 0:
                    ax.text(.5, index, "N/A", transform=ax.get_yaxis_transform(),
                            ha="center", va="center", color="#75858E", fontsize=7.5)
                    continue
                ax.errorbar(row.mean_difference, index,
                            xerr=np.array([[row.mean_difference-row.ci95_lower],
                                           [row.ci95_upper-row.mean_difference]]),
                            fmt="o", color="#0072B2", markersize=4.2,
                            markeredgecolor="white", markeredgewidth=.55,
                            elinewidth=1.05, capsize=2.4, capthick=.8, zorder=3)
            ax.axvline(0, color="#8195A0", linestyle=(0, (3, 2)), linewidth=.85, zorder=2)
            ax.set_xlim(*xlim)
            ax.set_ylim(len(BASELINES)-.45, -.6)
            ax.set_yticks(y, labels=BASELINES)
            ax.tick_params(axis="y", length=0, pad=7, labelsize=7.8)
            ax.tick_params(axis="x", length=2.5, pad=3, labelsize=7)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#A7B8C0")
            ax.grid(axis="x", color="#E2E9ED", linewidth=.5)
            ax.set_title("A  PCC difference" if j == 0 else "B  RMSE reduction",
                         loc="left", color="#193C51", fontsize=8.2, pad=9)
            ax.set_xlabel(r"$\Delta\mathrm{PCC}$" if j == 0 else r"$\Delta\mathrm{RMSE}$",
                          labelpad=6, fontsize=8)
        fig.text(.5, .055, "Positive values favor DynaGS", ha="center", va="center",
                 fontsize=7.7, color="#455560")
        fig.subplots_adjust(left=.19, right=.975, top=.90, bottom=.20, wspace=.22)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)
    return limits


if __name__ == "__main__":
    data = collect()
    pairs, summary = compare(data)
    DATA.mkdir(parents=True, exist_ok=True)
    for name, frame in [("runs", data), ("pairs", pairs), ("summary", summary)]:
        frame.to_csv(DATA / f"{STEM}_{name}.csv", index=False, encoding="utf-8-sig")
    limits = draw(summary)
    metadata = {"source_root": str(BASE.relative_to(ROOT)),
                "source_values": "Directly read from supplied 5.3 metrics.json overall fields; "
                                 "no recomputation from predictions or comparison with 5.2.",
                "condition": {"snps": 10000, "n_train": 400, "heritability": .8},
                "directions": {"pcc": "PCC_DynaGS minus PCC_baseline",
                               "rmse": "RMSE_baseline minus RMSE_DynaGS"},
                "point": "Arithmetic mean of 10 within-repeat paired differences.",
                "interval": "Percentile 95% paired bootstrap confidence interval; 10000 resamples "
                            "of the 10 repeat-level differences, seed 73. Identical resample "
                            "indices used across baselines and metrics.",
                "mean_baseline": "PCC is undefined and displayed as N/A; RMSE comparison uses 10 pairs.",
                "baseline_order": BASELINES, "axis_limits": limits,
                "figure": str(OUTPUT.relative_to(ROOT))}
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(OUTPUT)
