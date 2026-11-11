"""Saved 5.3 model trajectories and residuals for three fixed test individuals."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from dynags_figures.style import thesis_style
from figure_5_8_timepoint_model_performance import MODELS, STYLES

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
SELECTION = DATA / "figure_5_1_representative_predictions.json"
BASE = ROOT / "05_实验结果" / "5.3" / "P10000_NTR400_H0.8" / "rep_01"
STEM = "figure_5_9_individual_model_trajectories"
OUTPUT = ROOT / "07_论文图表" / "5-9_代表测试个体的模型预测轨迹与残差.png"


def collect() -> tuple[pd.DataFrame, dict]:
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    assert selection["condition"] == BASE.parent.name and selection["repeat"] == BASE.name
    high = next(person for person in selection["individuals"] if person["level"] == "High")
    original_id = high["sample_id"]
    original_index = high["sample_index_zero_based"]
    # The user requested another high-level example. Choose the next nearest
    # observed-level candidate, excluding the original individual, before reading errors.
    # The supplied NumPy archive stores its sample IDs as an object array.
    with np.load(BASE / "dynags/test_predictions.npz", allow_pickle=True) as archive:
        test_indices = archive["sample_idx"]
        sample_ids = archive["sample_id"]
        means = archive["target"].astype(float).mean(axis=1)
    target = float(np.quantile(means, .9))
    order = np.lexsort((test_indices, np.abs(means-target)))
    location = next(int(k) for k in order if test_indices[k] != original_index)
    high.update({"sample_id": str(sample_ids[location]),
                 "sample_index_zero_based": int(test_indices[location]),
                 "target_observed_mean": target, "selected_observed_mean": float(means[location])})
    selection["replacement"] = {
        "requested_change": "Another high-observed-level individual for visual inspection.",
        "original_sample_id": original_id, "original_sample_index": original_index,
        "new_sample_id": high["sample_id"], "new_sample_index": high["sample_index_zero_based"],
        "rule": "Nearest alternative to the 90th percentile of mean observed test phenotype, "
                "excluding the original high-level individual; ties use original row index."}
    rows = []
    reference = {}
    for model, folder in MODELS.items():
        source = BASE / folder / "test_predictions.npz"
        with np.load(source, allow_pickle=False) as archive:
            indices = archive["sample_idx"]
            times = archive["time_coordinates"].astype(float)
            prediction = archive["prediction"].astype(float)
            observed = archive["target"].astype(float)
        assert prediction.shape == observed.shape == (len(indices), len(times))
        assert np.isfinite(prediction).all() and np.isfinite(observed).all()
        for individual in selection["individuals"]:
            index = individual["sample_index_zero_based"]
            locations = np.flatnonzero(indices == index)
            assert len(locations) == 1
            location = int(locations[0])
            if index not in reference:
                reference[index] = (times.copy(), observed[location].copy())
            else:
                assert np.array_equal(times, reference[index][0])
                assert np.array_equal(observed[location], reference[index][1])
            for t, y, pred in zip(times, observed[location], prediction[location], strict=True):
                rows.append({"panel": individual["panel"], "level": individual["level"],
                             "sample_id": individual["sample_id"], "sample_index": index,
                             "model": model, "time": t, "observed": y, "prediction": pred,
                             "residual": pred-y, "source": str(source.relative_to(ROOT))})
    data = pd.DataFrame(rows)
    assert len(data) == 3 * 4 * 20
    assert not data.duplicated(["sample_index", "model", "time"]).any()
    return data, selection


def draw(data: pd.DataFrame) -> dict:
    upper_limit = float(np.ceil(data[["observed", "prediction"]].to_numpy().max()/5)*5)
    residual_limit = max(4., float(np.ceil(data.residual.abs().max()/.5)*.5))
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.8), sharex=True, sharey="row")
        for j, panel in enumerate("ABC"):
            part = data[data.panel == panel]
            observed = part[part.model == "DynaGS"].sort_values("time")
            for model in ["Cropformer", "Random Forest", "XGBoost", "DynaGS"]:
                frame = part[part.model == model].sort_values("time")
                for i, column in enumerate(["prediction", "residual"]):
                    axes[i, j].plot(frame.time, frame[column], **STYLES[model],
                                    lw=1.4 if model == "DynaGS" else 1., markersize=2.4,
                                    markerfacecolor="white", markeredgewidth=.65,
                                    markevery=[3, 7, 11, 15, 19],
                                    zorder=4 if model == "DynaGS" else 3)
            axes[0, j].scatter(observed.time, observed.observed, s=7.5, color="#7B8992",
                               edgecolors="white", linewidths=.25, zorder=5)
            axes[0, j].set_ylim(0, upper_limit)
            axes[0, j].set_yticks([0, 10, 20, 30])
            axes[0, j].set_title(f"{panel}  {part.level.iloc[0]} level ({part.sample_id.iloc[0]})",
                                 loc="left", fontsize=8, color="#193C51", pad=7)
            axes[1, j].axhline(0, color="#8DA0AC", lw=.7, linestyle=(0, (3, 2)), zorder=1)
            axes[1, j].set_ylim(-residual_limit, residual_limit)
            axes[1, j].set_yticks(np.linspace(-residual_limit, residual_limit, 5))
            axes[1, j].set_title(f"{'DEF'[j]}  Prediction residuals", loc="left",
                                 fontsize=8, color="#193C51", pad=7)
            for ax in axes[:, j]:
                ax.set_xlim(.5, 20.5)
                ax.set_xticks([1, 5, 10, 15, 20])
                ax.tick_params(length=2.5, pad=2, labelsize=7)
                ax.grid(axis="y", color="#DFE7EB", lw=.5)
                ax.spines[["left", "bottom"]].set_color("#A7B8C0")
        axes[0, 0].set_ylabel("Phenotype value", fontsize=8, labelpad=5)
        axes[1, 0].set_ylabel("Prediction − observed", fontsize=8, labelpad=5)
        fig.supxlabel("Time point", y=.115, fontsize=8)
        handles = [Line2D([], [], marker="o", ls="none", color="#7B8992", markersize=3,
                          label="Observed")]
        handles.extend(Line2D([], [], **STYLES[model], lw=1.4 if model == "DynaGS" else 1.,
                              markerfacecolor="white", markeredgewidth=.65, markersize=2.7,
                              label=model) for model in MODELS)
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .023),
                   ncol=5, fontsize=7, handlelength=1.9, handletextpad=.5, columnspacing=1.1)
        fig.subplots_adjust(left=.095, right=.985, top=.92, bottom=.205,
                            wspace=.17, hspace=.38)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)
    return {"phenotype": [0, upper_limit], "residual": [-residual_limit, residual_limit]}


if __name__ == "__main__":
    data, selection = collect()
    DATA.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA / f"{STEM}.csv", index=False, encoding="utf-8-sig")
    limits = draw(data)
    metadata = {"source_root": str(BASE.relative_to(ROOT)),
                "source_values": "Saved 5.3 test_predictions.npz prediction and target arrays, "
                                 "aligned by test sample index and time coordinate.",
                "selection_record": str(SELECTION.relative_to(ROOT)),
                "selection": "Retain the low and median individuals fixed for figure 5-1. "
                             "At the user's request, replace the high-level individual by the "
                             "nearest alternative to the 90th percentile of observed test means. "
                             "All plotted trajectories are read from 5.3.",
                "replacement": selection["replacement"],
                "individuals": selection["individuals"], "models": list(MODELS),
                "display": "Upper row: observed points and predicted values joined by straight "
                           "segments across all 20 saved time points. Lower row: predicted minus "
                           "observed values on the same original simulation scale. No smoothing.",
                "uncertainty": "Individual predictions from one fixed repeat; no interval bands.",
                "axis_limits": limits, "figure": str(OUTPUT.relative_to(ROOT))}
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Plotted rows:", len(data), "Individuals:", data.sample_id.unique().tolist())
    print(OUTPUT)
