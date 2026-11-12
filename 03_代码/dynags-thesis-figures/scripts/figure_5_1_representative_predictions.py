"""Plot the three user-fixed simulation examples synchronized with figure 5-9."""

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
from dynags_figures.labels import AXIS_LABELS

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
BASE = ROOT / "05_实验结果" / "5.2"
CONDITION = "P10000_NTR400_H0.8"
REPEAT = "rep_01"
RELATIVE = Path(CONDITION) / REPEAT
STEM = "figure_5_1_representative_predictions"
OUTPUT = ROOT / "07_论文图表" / "5-1_模拟测试个体的纵向表型预测轨迹.png"
INDIVIDUALS = [("A", "Low", "ID113"), ("B", "Middle", "ID477"), ("C", "High", "ID24")]


def prepare() -> tuple[pd.DataFrame, dict]:
    source = BASE / "experiments" / RELATIVE
    config = json.loads((source / "simulation_config.json").read_text(encoding="utf-8"))
    columns = config["time_columns"]
    time = np.asarray(config["time_coordinates"], dtype=float)
    observed = pd.read_csv(source / "sim_pheno.csv", dtype={"SampleID": str})
    truth = pd.read_csv(source / "truth" / "true_curve.csv", dtype={"SampleID": str})
    ids = pd.read_csv(source / "sample_ids.csv", dtype=str)["SampleID"].tolist()
    assert observed["SampleID"].tolist() == ids == truth["SampleID"].tolist()
    runs = {}
    for name, folder in [("oracle", "experiment1_oracle"),
                         ("screened", "experiment2_screened")]:
        with np.load(BASE / folder / RELATIVE / "test_predictions.npz") as archive:
            runs[name] = {k: archive[k] for k in
                          ["sample_idx", "time_coordinates", "predictions", "targets"]}
    indices = runs["oracle"]["sample_idx"]
    assert np.array_equal(indices, runs["screened"]["sample_idx"])
    test_observed = observed.iloc[indices][columns].to_numpy(float)
    for run in runs.values():
        assert np.array_equal(time, run["time_coordinates"])
        assert np.allclose(test_observed, run["targets"][:, 0, :], atol=1e-5)
        assert np.isfinite(run["predictions"]).all()
    levels = test_observed.mean(axis=1)
    initial_choices = []
    for panel, label, quantile in zip("ABC", ["Low", "Median", "High"], [.1, .5, .9]):
        target = float(np.quantile(levels, quantile))
        # Distance to the observed-level quantile, then original row index, breaks ties.
        k = int(np.lexsort((indices, np.abs(levels - target)))[0])
        i = int(indices[k])
        initial_choices.append({"panel": panel, "level": label, "sample_id": ids[i],
                                "sample_index_zero_based": i, "observed_level_quantile": quantile,
                                "target_observed_mean": target, "selected_observed_mean": float(levels[k])})
    choices = []
    rows = []
    for panel, label, sample_id in INDIVIDUALS:
        i = ids.index(sample_id)
        locations = np.flatnonzero(indices == i)
        assert len(locations) == 1
        k = int(locations[0])
        choices.append({"panel": panel, "level": label, "sample_id": sample_id,
                        "sample_index_zero_based": i, "selected_observed_mean": float(levels[k]),
                        "observed_mean_percentile_rank": float(100*np.mean(levels <= levels[k]))})
        for j, t in enumerate(time):
            rows.append({"condition": CONDITION, "repeat": REPEAT, "panel": panel,
                         "level": label, "sample_id": ids[i], "time": t,
                         "observed": test_observed[k, j],
                         "latent_truth": truth.iloc[i][columns[j]],
                         "oracle": runs["oracle"]["predictions"][k, 0, j],
                         "screened": runs["screened"]["predictions"][k, 0, j]})
    assert len({x["sample_id"] for x in choices}) == 3
    meta = {"condition": CONDITION, "repeat": REPEAT, "test_individual_count": len(indices),
            "selection": "User-fixed examples ID113, ID477 and ID24, matching figure 5-9. "
                         "ID113 came from the original low-level selection; ID24 was the nearest "
                         "alternative high-level example; ID477 was previously selected as a "
                         "DynaGS-favorable middle-level illustrative case in the 5.3 model comparison.",
            "initial_selection": {"condition": CONDITION, "repeat": REPEAT,
                                  "test_individual_count": len(indices),
                                  "selection": "Original quantile-based selection before user-requested replacements.",
                                  "individuals": initial_choices},
            "curve_display": "Saved values at observed time points, joined by straight segments; "
                             "no smoothing or interpolation of additional values.",
            "individuals": choices,
            "sources": [str((BASE / folder / RELATIVE / "test_predictions.npz").relative_to(ROOT))
                        for folder in ["experiment1_oracle", "experiment2_screened"]]
                       + [str((source / p).relative_to(ROOT)) for p in
                          ["sim_pheno.csv", "truth/true_curve.csv", "sample_ids.csv",
                           "simulation_config.json"]]}
    return pd.DataFrame(rows), meta


def draw(data: pd.DataFrame) -> None:
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.65), sharex=True, sharey=True)
        colors = {"truth": "#303C46", "oracle": "#0072B2", "screened": "#D55E00"}
        for ax, panel in zip(axes, "ABC"):
            part = data[data["panel"] == panel]
            t = part["time"].to_numpy()
            ax.plot(t, part["latent_truth"], color=colors["truth"], lw=1.5, zorder=3)
            ax.plot(t, part["oracle"], color=colors["oracle"], lw=1.35,
                    linestyle=(0, (4, 2)), zorder=4)
            ax.plot(t, part["screened"], color=colors["screened"], lw=1.35,
                    linestyle=(0, (4, 1.6, 1, 1.6)), zorder=5)
            ax.scatter(t, part["observed"], s=11, facecolors="white", edgecolors="#70818C",
                       linewidths=.65, zorder=6)
            ax.set_title(f"{panel}  {part['level'].iloc[0]} level ({part['sample_id'].iloc[0]})",
                         fontsize=8, loc="left", pad=7, color="#193C51")
            ax.set_xlim(.5, 20.5)
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.set_ylim(0, 35)
            ax.set_yticks([0, 10, 20, 30])
            ax.grid(axis="y", color="#DFE7EB", lw=.5)
            ax.spines[["left", "bottom"]].set_color("#A7B8C0")
            ax.tick_params(length=2.5, pad=2, labelsize=7)
        fig.supxlabel(AXIS_LABELS["time"], y=.19, fontsize=8)
        handles = [Line2D([], [], color=colors["truth"], lw=1.5, label="Latent truth"),
                   Line2D([], [], marker="o", ls="none", markerfacecolor="white",
                          markeredgecolor="#70818C", markersize=3.4, label="Observed"),
                   Line2D([], [], color=colors["oracle"], lw=1.35, ls=(0, (4, 2)),
                          label="Oracle SNPs"),
                   Line2D([], [], color=colors["screened"], lw=1.35,
                          ls=(0, (4, 1.6, 1, 1.6)), label="Screened SNPs")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .035),
                   ncol=4, fontsize=7.4, handlelength=2.6, columnspacing=1.9)
        fig.subplots_adjust(left=.105, right=.985, bottom=.34, top=.88, wspace=.16)
        plotting_area = axes[0].get_position()
        ylabel = fig.supylabel(AXIS_LABELS["phenotypic_value"], x=.018,
                              y=(plotting_area.y0 + plotting_area.y1) / 2,
                              va="center", fontsize=8)
        fig.canvas.draw()
        label_box = ylabel.get_window_extent(fig.canvas.get_renderer())
        axis_center = (axes[0].bbox.y0 + axes[0].bbox.y1) / 2
        if abs((label_box.y0 + label_box.y1) / 2 - axis_center) > .5:
            raise ValueError("Phenotypic value label is not centered on the plotting area")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    frame, metadata = prepare()
    destination = PROJECT / "data"
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / f"{STEM}.csv", index=False, encoding="utf-8-sig")
    (destination / f"{STEM}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    draw(frame)
    print(json.dumps(metadata["individuals"], ensure_ascii=False, indent=2))
    print(OUTPUT)
