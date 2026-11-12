"""Plot the nine pre-exported real maize OOF individual trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from dynags_figures.style import thesis_style
from dynags_figures.labels import AXIS_LABELS

SOURCE = ROOT / "05_实验结果/5.5/outputs/tfpp/maize"
DATA = PROJECT / "data"
STEM = "figure_5_13_real_individual_trajectories"
OUTPUT = ROOT / "07_论文图表/5-13_真实玉米代表个体的表型预测轨迹.png"
TRAITS = ["RGBVI", "CIVE", "IPCA"]
STRATA = ["low", "middle", "high"]
MODELS = ["DynaGS", "Cropformer"]
STYLES = {
    "Observed": {"color": "#606C76", "linestyle": "-", "marker": "o"},
    "DynaGS": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "Cropformer": {"color": "#AA4499", "linestyle": (0, (5, 1.7, 1, 1.7)), "marker": "D"},
}
FILES = ["analysis/representative_samples.csv", "analysis/representative_trajectories.csv",
         "protocol.json", "sample_ids.csv"]


def collect() -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    samples = pd.read_csv(SOURCE / FILES[0], dtype={"sample_id": "string"})
    trajectories = pd.read_csv(SOURCE / FILES[1], dtype={"sample_id": "string"})
    protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    ids = pd.read_csv(SOURCE / "sample_ids.csv", dtype={"SampleID": "string"})
    times = np.asarray(protocol["config"]["time_coordinates"], dtype=float)
    if len(samples) != 9 or len(trajectories) != 252:
        raise ValueError("Expected nine supplied individuals and 252 model-time records")
    if samples.duplicated(["trait", "stratum"]).any():
        raise ValueError("Duplicate representative sample within a trait-level combination")
    if trajectories.duplicated(["trait", "stratum", "model", "time"]).any():
        raise ValueError("Duplicate individual trajectory records")
    if set(trajectories.model) != set(MODELS) or set(trajectories.trait) != set(TRAITS):
        raise ValueError("Unexpected supplied models or traits")
    if not ((trajectories.suite == "main").all() and (trajectories.fraction == 1.).all()):
        raise ValueError("Expected the main joint-trait prediction experiment")
    if not np.isfinite(trajectories[["time", "observed", "predicted"]].to_numpy()).all():
        raise ValueError("Non-finite individual trajectory values")
    samples = samples.set_index(["trait", "stratum"])
    ordered = []
    split_sources = []
    for row, trait in enumerate(TRAITS):
        for col, stratum in enumerate(STRATA):
            sample = samples.loc[(trait, stratum)]
            part = trajectories[(trajectories.trait == trait) & (trajectories.stratum == stratum)]
            for field in ["repeat", "fold", "sample_idx", "sample_id"]:
                if part[field].nunique() != 1 or part[field].iloc[0] != sample[field]:
                    raise ValueError(f"Sample metadata differs between supplied files: {field}")
            reference = None
            for model in MODELS:
                frame = part[part.model == model].sort_values("time")
                np.testing.assert_array_equal(frame.time.to_numpy(), times)
                if reference is None:
                    reference = frame.observed.to_numpy()
                else:
                    np.testing.assert_array_equal(frame.observed.to_numpy(), reference)
            np.testing.assert_allclose(reference.mean(), sample.observed_mean, rtol=1e-6, atol=1e-8)
            index = int(sample.sample_idx)
            if str(ids.iloc[index].SampleID) != str(sample.sample_id):
                raise ValueError("Exported sample ID differs from the supplied row-ID mapping")
            split_path = f"shared/repeat_{int(sample['repeat']):02d}/fold_{int(sample.fold):02d}/split.npz"
            with np.load(SOURCE / split_path, allow_pickle=False) as split:
                if (index not in split["test_idx"] or index in split["train_idx"]
                        or index in split["val_idx"]):
                    raise ValueError("Representative trajectory is not from its stated outer test fold")
            split_sources.append(split_path)
            ordered.append({"panel": chr(ord("A") + row * 3 + col), "trait": trait,
                            "stratum": stratum, **sample.to_dict(),
                            "observed_mean_from_exported_times": float(reference.mean()),
                            "outer_test_split": split_path})
    samples = pd.DataFrame(ordered)
    labels = samples[["panel", "trait", "stratum"]]
    trajectories = trajectories.merge(labels, on=["trait", "stratum"], validate="many_to_one")
    trajectories = trajectories.sort_values(["panel", "model", "time"]).reset_index(drop=True)
    return samples, trajectories, protocol, sorted(set(split_sources))


def draw(samples: pd.DataFrame, data: pd.DataFrame, protocol: dict) -> dict:
    times = np.asarray(protocol["config"]["time_coordinates"], dtype=float)
    limits = {}
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(3, 3, figsize=(6.3, 5.65), sharex=True, sharey="row")
        fig.subplots_adjust(left=.105, right=.98, top=.875, bottom=.13,
                            wspace=.20, hspace=.53)
        for row, trait in enumerate(TRAITS):
            trait_data = data[data.trait == trait]
            lo = float(trait_data[["observed", "predicted"]].to_numpy().min())
            hi = float(trait_data[["observed", "predicted"]].to_numpy().max())
            pad = .075 * (hi - lo)
            ylim = [lo - pad, hi + pad]
            limits[trait] = ylim
            for col, stratum in enumerate(STRATA):
                ax = axes[row, col]
                sample = samples[(samples.trait == trait) & (samples.stratum == stratum)].iloc[0]
                part = trait_data[trait_data.stratum == stratum]
                observed = part[part.model == "DynaGS"].sort_values("time")
                line, = ax.plot(observed.time, observed.observed, **STYLES["Observed"],
                                linewidth=.9, markersize=2.25, markeredgewidth=.25,
                                markeredgecolor="white", zorder=2)
                np.testing.assert_array_equal(line.get_xdata(), times)
                np.testing.assert_array_equal(line.get_ydata(), observed.observed.to_numpy())
                for model in ["Cropformer", "DynaGS"]:
                    frame = part[part.model == model].sort_values("time")
                    line, = ax.plot(frame.time, frame.predicted, **STYLES[model],
                                    linewidth=1.35 if model == "DynaGS" else 1.05,
                                    markersize=3.0 if model == "DynaGS" else 2.65,
                                    markerfacecolor="white", markeredgewidth=.65,
                                    markevery=[0, 4, 8, 11, 13],
                                    zorder=4 if model == "DynaGS" else 3)
                    np.testing.assert_array_equal(line.get_xdata(), times)
                    np.testing.assert_array_equal(line.get_ydata(), frame.predicted.to_numpy())
                ax.set_ylim(*ylim)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
                formatter = ScalarFormatter(useOffset=False)
                formatter.set_scientific(False)
                ax.yaxis.set_major_formatter(formatter)
                ax.set_xlim(7, 104)
                ax.set_xticks([10, 30, 50, 75, 100])
                ax.tick_params(length=2.5, pad=2.5, labelsize=7, width=.65)
                ax.grid(axis="y", color="#E0E7EC", linewidth=.5)
                ax.spines[["left", "bottom"]].set_color("#9EAFB9")
                ax.spines[["left", "bottom"]].set_linewidth(.7)
                if lo < 0 < hi:
                    ax.axhline(0, color="#9BAAB4", linestyle=(0, (3, 2)), linewidth=.65, zorder=0)
                ax.set_title(f"{sample.panel}  Sample {sample.sample_id}", loc="left",
                             fontsize=8, color="#193C51", pad=6)
                if col == 0:
                    ax.set_ylabel(trait, fontsize=8.5, labelpad=7)

        for col, heading in enumerate(["Low level", "Middle level", "High level"]):
            bounds = axes[0, col].get_position()
            fig.text(bounds.x0 + bounds.width / 2, .932, heading, ha="center", va="center",
                     fontsize=8.5, color="#193C51", fontweight="bold")
        handles = [Line2D([], [], **STYLES["Observed"], linewidth=.9, markersize=3,
                          markeredgecolor="white", markeredgewidth=.25, label="Observed")]
        handles.extend(Line2D([], [], **STYLES[model], linewidth=1.35 if model == "DynaGS" else 1.05,
                              markersize=3, markerfacecolor="white", markeredgewidth=.65,
                              label=model) for model in MODELS)
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .997),
                   ncol=3, fontsize=7.6, handlelength=2.5, handletextpad=.6, columnspacing=1.8)
        fig.text(.54, .068, AXIS_LABELS["time"], ha="center", va="center", fontsize=8.3)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_items = list(fig.texts) + fig.legends[0].get_texts()
        for ax in axes.flat:
            text_items += [ax._left_title, ax.yaxis.label]
            text_items += [label for loc, label in zip(ax.get_xticks(), ax.get_xticklabels())
                           if ax.get_xlim()[0] <= loc <= ax.get_xlim()[1]]
            text_items += [label for loc, label in zip(ax.get_yticks(), ax.get_yticklabels())
                           if ax.get_ylim()[0] <= loc <= ax.get_ylim()[1]]
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
    samples, trajectories, protocol, splits = collect()
    limits = draw(samples, trajectories, protocol)
    DATA.mkdir(parents=True, exist_ok=True)
    samples.to_csv(DATA / f"{STEM}_samples.csv", index=False, encoding="utf-8-sig")
    trajectories.to_csv(DATA / f"{STEM}_trajectories.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_root": str(SOURCE.relative_to(ROOT)),
        "source_files_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                                for name in FILES + splits},
        "source_values": "Directly plotted the supplied observed and predicted values from representative_trajectories.csv.",
        "sample_selection": "Used all nine individuals already listed in representative_samples.csv, preserving supplied low/middle/high strata and sample identities.",
        "sample_id_meaning": "The supplied row-index sample IDs, displayed as Sample plus the exported ID.",
        "checks": "Same sample IDs, repeat, outer fold, all fourteen actual times and identical observed values across models; outer-test membership and row-ID mapping checked.",
        "layout": "Three rows (RGBVI, CIVE, IPCA) by three columns (low, middle, high observed level); shared y limits within each trait.",
        "models": MODELS, "styles": STYLES,
        "times": protocol["config"]["time_coordinates"],
        "n_representative_samples": len(samples), "n_trajectory_records": len(trajectories),
        "line": "Straight segments connecting all fourteen supplied time-point values for each single individual; no smoothing or averaging.",
        "markers": "All observed points displayed; model markers at indices 0, 4, 8, 11, 13 while model lines include all fourteen values.",
        "axis_limits": limits, "figure_size_inches": [6.3, 5.65], "dpi": 600,
        "figure": str(OUTPUT.relative_to(ROOT)),
        "caption": "图 5-13 真实玉米代表个体的表型预测轨迹",
        "caption_note": "每行依次为 RGBVI、CIVE 和 IPCA，每列依次为已提供代表样本清单中的低、中、高观测水平个体。灰色曲线为观测值，蓝色和紫色曲线分别为 DynaGS 与 Cropformer 的折外预测。各面板保留实际观测时间和性状原始尺度，同一性状采用相同纵轴范围。面板中的 Sample 编号沿用结果文件中的样本编号。",
    }
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(OUTPUT), "samples": len(samples),
                      "trajectory_records": len(trajectories), "verified_outer_test_splits": len(splits)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
