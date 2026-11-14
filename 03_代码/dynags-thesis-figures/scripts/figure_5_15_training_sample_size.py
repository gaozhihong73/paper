"""Plot supplied DynaGS learning curves on a fixed outer test fold per repeat."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator, ScalarFormatter
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from dynags_figures.style import thesis_style
from dynags_figures.labels import AXIS_LABELS, validate_figure_axis_labels

SOURCE = ROOT / "05_实验结果/5.5/outputs/tfpp/maize"
DATA = PROJECT / "data"
STEM = "figure_5_15_training_sample_size"
OUTPUT = ROOT / "07_论文图表/5-15_不同训练样本规模下的真实玉米性状预测表现.png"
FRACTIONS = [.25, .50, .75, 1.]
TRAIN_COUNTS = [72, 144, 216, 288]
METRICS = ["pcc", "r2", "rmse"]
FILES = ["protocol.json", "analysis/summary_protocol.json", "analysis/repeat_metrics.csv",
         "analysis/summary.csv", "analysis/fold_metrics.csv", "split_counts.csv"]
BLUE = "#0072B2"
GREY = "#8396A2"


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[str]]:
    protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    traits = protocol["config"]["phenotype_names"]
    repeats = list(range(1, len(protocol["protocol"]["seeds"]) + 1))
    if len(traits) != 8 or repeats != [1, 2, 3, 4, 5]:
        raise ValueError("Expected eight traits and five repeats")

    def read(name):
        return pd.read_csv(SOURCE / "analysis" / name, dtype={"time": "string"})

    runs = read("repeat_metrics.csv")
    runs = runs[(runs.suite == "size") & (runs.model == "DynaGS") & (runs.time == "all")
                & runs.metric.isin(METRICS)].copy()
    source_summary = read("summary.csv")
    source_summary = source_summary[(source_summary.suite == "size")
                                    & (source_summary.model == "DynaGS")
                                    & (source_summary.time == "all")
                                    & source_summary.metric.isin(METRICS)]
    keys = ["fraction", "trait", "metric"]
    if len(runs) != 480 or len(source_summary) != 96:
        raise ValueError("Unexpected learning-curve record counts")
    if runs.duplicated(keys + ["repeat"]).any() or source_summary.duplicated(keys).any():
        raise ValueError("Duplicate learning-curve records")
    if set(runs.fraction) != set(FRACTIONS) or set(runs.trait) != set(traits):
        raise ValueError("Unexpected training fractions or traits")
    if not ((runs.n_test == 80).all() and runs.valid.all() and np.isfinite(runs.value).all()):
        raise ValueError("Expected finite metrics on 80 test individuals per repeat")

    count_source = pd.read_csv(SOURCE / "split_counts.csv")
    split_rows, split_sources = [], []
    for repeat in repeats:
        full_path = f"shared/repeat_{repeat:02d}/fold_01/split.npz"
        with np.load(SOURCE / full_path, allow_pickle=False) as archive:
            full = {key: archive[key].copy() for key in ["train_idx", "val_idx", "test_idx"]}
        for fraction, n_train in zip(FRACTIONS, TRAIN_COUNTS, strict=True):
            condition = "main" if fraction == 1 else f"size_{int(fraction * 100):03d}"
            split_file = full_path if fraction == 1 else f"shared/repeat_{repeat:02d}/{condition}/split.npz"
            with np.load(SOURCE / split_file, allow_pickle=False) as archive:
                train, val, test = [archive[key] for key in ["train_idx", "val_idx", "test_idx"]]
                np.testing.assert_array_equal(np.sort(val), np.sort(full["val_idx"]))
                np.testing.assert_array_equal(np.sort(test), np.sort(full["test_idx"]))
                if len(train) != n_train or len(val) != 32 or len(test) != 80:
                    raise ValueError("Actual split counts differ from the training-size labels")
                if len(np.unique(train)) != n_train or not np.isin(train, full["train_idx"]).all():
                    raise ValueError("Training subset differs from the main fold training pool")
                if np.intersect1d(train, test).size or np.intersect1d(train, val).size:
                    raise ValueError("Training subset overlaps validation or test individuals")
            count = count_source[(count_source["repeat"] == repeat) & (count_source.fold == 1)
                                 & (count_source.condition == condition)]
            if len(count) != 1:
                raise ValueError("Missing or duplicate split-count record")
            np.testing.assert_array_equal(count[["n_train", "n_validation", "n_test"]].iloc[0], [n_train, 32, 80])
            split_rows.append({"repeat": repeat, "fraction": fraction, "n_train": n_train,
                               "n_validation": 32, "n_test": 80, "outer_fold": 1,
                               "split_file": split_file})
            split_sources.append(split_file)
    split_counts = pd.DataFrame(split_rows)
    runs = runs.merge(split_counts.drop(columns="n_test"), on=["repeat", "fraction"], validate="many_to_one")

    full_metrics = read("fold_metrics.csv")
    full_metrics = full_metrics[(full_metrics.suite == "main") & (full_metrics.model == "DynaGS")
                                & (full_metrics.fold == 1) & (full_metrics.time == "all")
                                & full_metrics.metric.isin(METRICS)]
    full_comparison = runs[runs.fraction == 1].merge(full_metrics,
                                                   on=["repeat", "trait", "metric"],
                                                   suffixes=("_size", "_main"), validate="one_to_one")
    if len(full_comparison) != 120:
        raise ValueError("Incomplete 100% reference condition")
    np.testing.assert_allclose(full_comparison.value_size, full_comparison.value_main, rtol=0, atol=1e-12)

    source_summary = source_summary.set_index(keys)
    summary_rows = []
    for key, group in runs.groupby(keys, sort=False):
        group = group.sort_values("repeat")
        if group["repeat"].tolist() != repeats:
            raise ValueError("Incomplete repeat set at a training size")
        source = source_summary.loc[key]
        np.testing.assert_allclose([source["mean"], source["std"]],
                                   [group.value.mean(), group.value.std(ddof=1)], rtol=0, atol=1e-12)
        raw = json.loads(source.raw_repeats)
        np.testing.assert_allclose([raw[str(i)] for i in repeats], group.value, rtol=0, atol=1e-12)
        if int(source.n_valid) != 5 or group.n_train.nunique() != 1:
            raise ValueError("Unexpected summary repeat count or training size")
        summary_rows.append(dict(zip(keys, key, strict=True)) | {
            "n_train": int(group.n_train.iloc[0]), "mean": source["mean"],
            "sample_sd": source["std"], "band_lower": source["mean"] - source["std"],
            "band_upper": source["mean"] + source["std"], "n_repeats": 5,
            "n_validation_per_repeat": 32, "n_test_per_repeat": 80, "outer_fold": 1})
    return runs, pd.DataFrame(summary_rows), split_counts, protocol, sorted(set(split_sources))


def draw(runs: pd.DataFrame, summary: pd.DataFrame, traits: list[str]) -> dict:
    limits = {}
    pcc_summary = summary[summary.metric == "pcc"]
    pcc_raw = runs[runs.metric == "pcc"].value
    pcc_limits = (float(np.floor(min(pcc_raw.min(), pcc_summary.band_lower.min()) * 10) / 10),
                  float(np.ceil(max(pcc_raw.max(), pcc_summary.band_upper.max()) * 10) / 10))
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig = plt.figure(figsize=(6.3, 7.6))
        blocks = {
            "pcc": fig.add_gridspec(2, 4, left=.085, right=.98, bottom=.565, top=.89,
                                     hspace=.48, wspace=.48),
            "rmse": fig.add_gridspec(2, 4, left=.085, right=.98, bottom=.115, top=.44,
                                      hspace=.48, wspace=.48),
        }
        axes = []
        for metric in ["pcc", "rmse"]:
            for index, trait in enumerate(traits):
                ax = fig.add_subplot(blocks[metric][index // 4, index % 4])
                axes.append(ax)
                part = summary[(summary.metric == metric) & (summary.trait == trait)].sort_values("n_train")
                raw = runs[(runs.metric == metric) & (runs.trait == trait)]
                np.testing.assert_array_equal(part.n_train.to_numpy(), TRAIN_COUNTS)
                ax.fill_between(part.n_train, part.band_lower, part.band_upper,
                                color=BLUE, alpha=.12, linewidth=0, zorder=1)
                ax.scatter(raw.n_train, raw.value, s=7.5, color=GREY, alpha=.65,
                           edgecolors="white", linewidths=.2, zorder=3)
                ax.plot(part.n_train, part["mean"], color=BLUE, linewidth=1.45,
                        marker="o", markersize=3.2, markerfacecolor="white",
                        markeredgewidth=.8, zorder=4)
                if metric == "pcc":
                    ylim = pcc_limits
                    ax.yaxis.set_major_locator(MultipleLocator(.2))
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                else:
                    low = min(float(raw.value.min()), float(part.band_lower.min()))
                    high = max(float(raw.value.max()), float(part.band_upper.max()))
                    span = high - low
                    ylim = (max(0., low - .12 * span), high + .15 * span)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_powerlimits((-2, 3))
                    formatter.set_useOffset(False)
                    ax.yaxis.set_major_formatter(formatter)
                    offset = ax.yaxis.get_offset_text()
                    offset.set_x(1.)
                    offset.set_horizontalalignment("right")
                    offset.set_fontsize(6.8)
                ax.set_ylim(*ylim)
                limits[f"{metric}/{trait}"] = list(ylim)
                ax.set_xlim(60, 300)
                ax.set_xticks(TRAIN_COUNTS)
                ax.tick_params(length=2.3, pad=2, labelsize=6.9, width=.65)
                ax.grid(axis="y", color="#E0E7EC", linewidth=.45)
                ax.spines[["left", "bottom"]].set_color("#9EAFB9")
                ax.spines[["left", "bottom"]].set_linewidth(.65)
                letter = chr(ord("A") + index + (0 if metric == "pcc" else 8))
                ax.set_title(f"{letter}  {trait}", loc="left", fontsize=8.1,
                             color="#193C51", pad=6)
        handles = [Line2D([], [], marker="o", linestyle="none", color=GREY,
                          markersize=3.2, alpha=.75, label="Repeat"),
                   Line2D([], [], color=BLUE, marker="o", markerfacecolor="white",
                          markeredgewidth=.8, markersize=3.4, linewidth=1.45, label="Mean"),
                   Patch(facecolor=BLUE, alpha=.12, edgecolor="none", label="±1 SD")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .997),
                   ncol=3, fontsize=7.6, handlelength=2.5, handletextpad=.6, columnspacing=1.8)
        fig.text(.53, .935, AXIS_LABELS["pcc"], ha="center", va="center", fontsize=8.8,
                 color="#193C51", fontweight="bold")
        fig.text(.53, .518, AXIS_LABELS["training_sample_size"], ha="center", va="center", fontsize=8)
        fig.text(.53, .478, AXIS_LABELS["rmse"], ha="center", va="center", fontsize=8.8,
                 color="#193C51", fontweight="bold")
        fig.text(.53, .053, AXIS_LABELS["training_sample_size"], ha="center", va="center", fontsize=8)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        validate_figure_axis_labels(fig)
        text_items = list(fig.texts) + fig.legends[0].get_texts()
        for ax in axes:
            offset = ax.yaxis.get_offset_text()
            text_items += [ax._left_title, offset] + ax.get_xticklabels()
            text_items += [label for value, label in zip(ax.get_yticks(), ax.get_yticklabels())
                           if ax.get_ylim()[0] <= value <= ax.get_ylim()[1]]
            if offset.get_text() and offset.get_window_extent(renderer).overlaps(ax._left_title.get_window_extent(renderer)):
                raise ValueError("Scientific multiplier overlaps a panel title")
        for item in text_items:
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
    runs, summary, counts, protocol, split_sources = collect()
    limits = draw(runs, summary, protocol["config"]["phenotype_names"])
    DATA.mkdir(parents=True, exist_ok=True)
    for suffix, frame in [("runs", runs), ("summary", summary), ("splits", counts)]:
        frame.to_csv(DATA / f"{STEM}_{suffix}.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_root": str(SOURCE.relative_to(ROOT)),
        "source_files_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                                for name in FILES + split_sources},
        "source_filter": {"suite": "size", "model": "DynaGS", "time": "all"},
        "source_values": "Supplied repeat-level metrics and summaries, used without prediction-based recomputation.",
        "checks": "Means, sample SD, raw_repeats and repeat counts verified; validation/test indices match the first main outer fold within each repeat; training subsets and actual counts verified.",
        "reference_100_percent": "Verified against main DynaGS fold-1 metrics for all five repeats, eight traits and three metrics.",
        "fractions": FRACTIONS, "training_sample_sizes": TRAIN_COUNTS,
        "validation_samples_per_repeat": 32, "test_samples_per_repeat": 80,
        "outer_fold": 1, "n_repeats": 5,
        "traits": protocol["config"]["phenotype_names"],
        "displayed_metrics": ["pcc", "rmse"], "saved_metrics": METRICS,
        "line": "Arithmetic mean of five provided repeat-level values at each of four training sizes.",
        "band": "Mean plus or minus one sample SD across five repeats, ddof=1.",
        "points": "All five original repeat values at each actual training count; no jitter.",
        "rmse": "Original scale within each trait, with separate y ranges and standard scientific tick multipliers.",
        "axis_limits": limits, "figure_size_inches": [6.3, 7.6], "dpi": 600,
        "figure": str(OUTPUT.relative_to(ROOT)),
        "caption": "图 5-15 不同训练样本规模下的真实玉米性状预测表现",
        "caption_note": "A—H：八个性状的 PCC；I—P：对应性状的 RMSE。训练人数依次为 72、144、216、288，对应完整拟合训练集的 25%、50%、75%、100%。每次重复固定第 1 外层折的 32 个验证样本和 80 个测试样本，100% 结果复用该外层折的主实验结果。灰色散点表示五次重复的原始值，蓝色曲线表示均值，阴影表示均值上下一个样本标准差（ddof=1）。RMSE 保留各性状原始尺度。",
    }
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(OUTPUT), "n_runs": len(runs), "n_summary_rows": len(summary),
                      "n_split_conditions": len(counts), "training_counts": TRAIN_COUNTS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
