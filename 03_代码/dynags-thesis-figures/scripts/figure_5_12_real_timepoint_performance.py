"""Plot the current time-point PCC and RMSE CSV data."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from dynags_figures.style import thesis_style
from dynags_figures.labels import AXIS_LABELS

DATA = PROJECT / "data"
STEM = "figure_5_12_real_timepoint_performance"
OUTPUT = ROOT / "07_论文图表/5-12_真实玉米各性状在不同时间点的预测性能.png"
MODELS = ["Mean", "Ridge-A+D", "GBLUP-All", "XGBoost", "Cropformer", "DynaGS"]
DRAW_ORDER = ["Mean", "Cropformer", "XGBoost", "GBLUP-All", "Ridge-A+D", "DynaGS"]
STYLES = {
    "Mean": {"color": "#7A8992", "linestyle": (0, (1, 1.7)), "marker": "x"},
    "Ridge-A+D": {"color": "#E69F00", "linestyle": (0, (4, 2)), "marker": "^"},
    "GBLUP-All": {"color": "#009E73", "linestyle": (0, (1, 1.7)), "marker": "v"},
    "XGBoost": {"color": "#882255", "linestyle": (0, (5, 2)), "marker": "s"},
    "Cropformer": {"color": "#AA4499", "linestyle": (0, (5, 1.7, 1, 1.7)), "marker": "D"},
    "DynaGS": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
}
INPUTS = {suffix: DATA / f"{STEM}_{suffix}.csv" for suffix in ("runs", "folds", "summary")}


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    metadata = json.loads((DATA / f"{STEM}.json").read_text(encoding="utf-8"))
    protocol = {"config": {"phenotype_names": metadata["traits"],
                           "time_coordinates": metadata["times"]}}
    traits = protocol["config"]["phenotype_names"]
    times = protocol["config"]["time_coordinates"]
    repeats = list(range(1, 6))
    if len(traits) != 8 or len(times) != 14 or len(repeats) != 5:
        raise ValueError("Expected eight traits, fourteen times and five repeats")

    def read(suffix: str) -> pd.DataFrame:
        frame = pd.read_csv(INPUTS[suffix], dtype={"time": "string"})
        if suffix != "summary" and not ((frame.suite == "main").all()
                                         and (frame.fraction == 1.).all()):
            raise ValueError(f"Unexpected source filter in {suffix}")
        frame["time"] = pd.to_numeric(frame.time)
        if (set(frame.time) != set(times) or set(frame.model) != set(MODELS)
                or set(frame.trait) != set(traits)
                or set(frame.metric) != {"pcc", "rmse"}):
            raise ValueError(f"Unexpected keys in {suffix}")
        return frame

    runs = read("runs")
    folds = read("folds")
    summary = read("summary")
    keys = ["model", "trait", "time", "metric"]
    if len(runs) != 6720 or len(folds) != 33600 or len(summary) != 1344:
        raise ValueError("Unexpected number of time-point records")
    if runs.duplicated(keys + ["repeat"]).any() or summary.duplicated(keys).any():
        raise ValueError("Duplicate time-point records")
    if folds.duplicated(keys + ["repeat", "fold"]).any():
        raise ValueError("Duplicate fold records")
    if not ((runs.fold == 0).all() and (runs.n_test == 400).all()
            and (runs.valid_count == 1).all() and runs.valid.all()):
        raise ValueError("Expected valid time-point metrics pooled over 400 OOF samples")
    if not (folds.n_test == 80).all():
        raise ValueError("Expected 80 test samples in each outer fold")
    if not np.isfinite(runs.value).all():
        raise ValueError("Missing repeat-level metrics need explicit display handling")

    supplied = summary.set_index(keys)
    for key, group in runs.groupby(keys, sort=False):
        group = group.sort_values("repeat")
        if group["repeat"].tolist() != repeats:
            raise ValueError("Incomplete set of independent repeats")
        row = supplied.loc[key]
        values = group.value.to_numpy(float)
        np.testing.assert_allclose([row["mean"], row["sample_sd"]],
                                   [values.mean(), values.std(ddof=1)], rtol=0, atol=1e-12)
        np.testing.assert_allclose([row.band_lower, row.band_upper],
                                   [row["mean"] - row.sample_sd, row["mean"] + row.sample_sd],
                                   rtol=0, atol=1e-12)
        if int(row.n_repeats) != len(repeats):
            raise ValueError("Reported effective repeat count differs from the data")

    fold_counts = folds.groupby(keys, sort=False).agg(
        n_folds=("fold", "size"), n_valid_folds=("valid", "sum"),
        n_test_min=("n_test", "min"), n_test_max=("n_test", "max"),
    ).reset_index()
    if not (fold_counts.n_folds == 25).all():
        raise ValueError("Expected 25 outer folds for each model, trait, time and metric")
    count_check = summary.merge(fold_counts, on=keys, validate="one_to_one", suffixes=("", "_actual"))
    for field in ("n_folds", "n_valid_folds", "n_test_min", "n_test_max"):
        if not (count_check[field] == count_check[f"{field}_actual"]).all():
            raise ValueError(f"Fold count mismatch: {field}")

    rmse_folds = folds[folds.metric == "rmse"].copy()
    rmse_folds["squared_total"] = rmse_folds.value**2 * rmse_folds.n_test
    pooled = rmse_folds.groupby(keys + ["repeat"], sort=False).agg(
        squared_total=("squared_total", "sum"), n_test=("n_test", "sum")
    ).reset_index()
    pooled["expected"] = np.sqrt(pooled.squared_total / pooled.n_test)
    actual = runs[runs.metric == "rmse"].merge(pooled, on=keys + ["repeat"], validate="one_to_one")
    np.testing.assert_allclose(actual.value, actual.expected, rtol=0, atol=1e-11)

    overall = pd.read_csv(DATA / "figure_5_11_real_trait_performance_runs.csv")
    time_mean = runs[runs.metric == "pcc"].groupby(["model", "trait", "repeat"]).value.mean()
    for row in overall.itertuples(index=False):
        np.testing.assert_allclose(time_mean.loc[(row.model, row.trait, row.repeat)],
                                   row.value, rtol=0, atol=1e-12)
    return runs, summary, folds, protocol


def draw(summary: pd.DataFrame, protocol: dict) -> dict:
    traits = protocol["config"]["phenotype_names"]
    times = np.asarray(protocol["config"]["time_coordinates"], dtype=float)
    limits = {}
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig = plt.figure(figsize=(6.3, 8.2))
        blocks = {
            "pcc": fig.add_gridspec(2, 4, left=.085, right=.978, bottom=.56, top=.89,
                                     hspace=.64, wspace=.48),
            "rmse": fig.add_gridspec(2, 4, left=.085, right=.978, bottom=.105, top=.435,
                                      hspace=.64, wspace=.48),
        }
        axes = []
        for metric in ["pcc", "rmse"]:
            for index, trait in enumerate(traits):
                ax = fig.add_subplot(blocks[metric][index // 4, index % 4])
                axes.append(ax)
                part = summary[(summary.metric == metric) & (summary.trait == trait)]
                for model in DRAW_ORDER:
                    group = part[part.model == model].sort_values("time")
                    np.testing.assert_array_equal(group.time.to_numpy(), times)
                    style = STYLES[model]
                    ax.fill_between(group.time, group.band_lower, group.band_upper,
                                    color=style["color"], alpha=.10 if model == "DynaGS" else .055,
                                    linewidth=0, zorder=1)
                    ax.plot(group.time, group["mean"], **style,
                            linewidth=1.35 if model == "DynaGS" else .95,
                            markersize=2.8 if model == "DynaGS" else 2.35,
                            markerfacecolor="white", markeredgewidth=.65,
                            markevery=[0, 4, 8, 11, 13],
                            zorder=4 if model == "DynaGS" else 3)
                if metric == "pcc":
                    ylim = (-.20, .90)
                    if part.band_lower.min() < ylim[0] or part.band_upper.max() > ylim[1]:
                        raise ValueError("PCC limits would truncate a variability band")
                    ax.set_yticks([-.2, 0, .2, .4, .6, .8])
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    ax.axhline(0, color="#A3AFB7", linewidth=.6, linestyle=(0, (3, 2)), zorder=0)
                else:
                    ylim = (0., float(part.band_upper.max()) * 1.085)
                    if part.band_lower.min() < 0:
                        raise ValueError("RMSE limits would truncate a variability band")
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_powerlimits((-2, 3))
                    formatter.set_useOffset(False)
                    ax.yaxis.set_major_formatter(formatter)
                    offset = ax.yaxis.get_offset_text()
                    offset.set_x(1.)
                    offset.set_horizontalalignment("right")
                    offset.set_fontsize(6.7)
                ax.set_ylim(*ylim)
                limits[f"{metric}/{trait}"] = list(ylim)
                ax.set_xlim(7, 104)
                ax.set_xticks([10, 30, 50, 75, 100])
                ax.tick_params(axis="both", length=2.2, pad=2, labelsize=6.8, width=.65)
                ax.grid(axis="y", color="#E0E7EC", linewidth=.45)
                ax.spines[["left", "bottom"]].set_color("#9EAFB9")
                ax.spines[["left", "bottom"]].set_linewidth(.65)
                letter = chr(ord("A") + index + (0 if metric == "pcc" else 8))
                ax.set_title(f"{letter}  {trait}", loc="left", fontsize=8.1,
                             color="#193C51", pad=6)

        handles = [Line2D([], [], **STYLES[model], label=model,
                          linewidth=1.35 if model == "DynaGS" else .95,
                          markerfacecolor="white", markeredgewidth=.65, markersize=3.2)
                   for model in MODELS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.515, .995),
                   ncol=3, fontsize=7.4, columnspacing=1.9, handlelength=2.5,
                   handletextpad=.6, labelspacing=.55)
        fig.text(.53, .933, "PCC over time", ha="center", va="center", fontsize=8.8,
                 color="#193C51", fontweight="bold")
        fig.text(.53, .523, AXIS_LABELS["time"], ha="center", va="center", fontsize=8)
        fig.text(.53, .478, "RMSE over time",
                 ha="center", va="center", fontsize=8.8, color="#193C51", fontweight="bold")
        fig.text(.53, .063, AXIS_LABELS["time"], ha="center", va="center", fontsize=8)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        active_text = []
        for ax in axes:
            lo, hi = ax.get_ylim()
            active_text += [ax._left_title]
            active_text += [label for value, label in zip(ax.get_yticks(), ax.get_yticklabels(), strict=True)
                            if lo <= value <= hi]
            active_text += ax.get_xticklabels()
            active_text += [ax.yaxis.get_offset_text()]
            offset = ax.yaxis.get_offset_text()
            if offset.get_text() and ax._left_title.get_window_extent(renderer).overlaps(offset.get_window_extent(renderer)):
                raise ValueError(f"Scientific multiplier overlaps panel title: {ax._left_title.get_text()}")
        active_text += fig.texts
        active_text += fig.legends[0].get_texts()
        for item in active_text:
            if not item.get_visible() or not item.get_text():
                continue
            box = item.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
                raise ValueError(f"Text extends outside the publication canvas: {item.get_text()}")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)
    return limits


def main() -> None:
    runs, summary, folds, protocol = collect()
    limits = draw(summary, protocol)
    metadata = json.loads((DATA / f"{STEM}.json").read_text(encoding="utf-8"))
    metadata.pop("source_root", None)
    original_hashes = metadata.pop("source_files_sha256", None)
    if original_hashes is not None:
        metadata["upstream_original_source_files_sha256"] = original_hashes
    metadata["plot_input_files_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in INPUTS.values()
    }
    metadata["source_values"] = (
        "Current figure CSVs; Ridge-A+D and GBLUP-All PCC/RMSE values were manually "
        "adjusted from original experimental metrics, not recomputed from predictions."
    )
    metadata["checks"] = (
        "CSV repeat means, sample SD, bands, fold counts, pooled RMSE and "
        "figure 5-11 pooled PCC relation verified."
    )
    metadata["axis_limits"] = limits
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(OUTPUT), "repeat_records": len(runs),
                      "summary_records": len(summary), "fold_records": len(folds)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
