"""Plot DynaGS OOF trajectories for eight maize traits in an 8-by-3 layout."""

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
from dynags_figures.labels import AXIS_LABELS, validate_figure_axis_labels

SOURCE = ROOT / "05_实验结果/5.5/outputs/tfpp/maize"
DATA = PROJECT / "data"
STEM = "figure_5_13_real_individual_trajectories"
OUTPUT = ROOT / "07_论文图表/5-13_真实玉米代表个体的表型预测轨迹.png"
REPEAT = 1
FIGURE_SIZE = (6.3, 8.85)
STYLES = {
    "Observed": {"color": "#606C76", "linestyle": "-", "marker": "o"},
    "DynaGS": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
}


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[str]]:
    protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    traits = protocol["config"]["phenotype_names"]
    times = np.asarray(protocol["config"]["time_coordinates"], dtype=float)
    ids = pd.read_csv(SOURCE / "sample_ids.csv", dtype={"SampleID": "string"}).SampleID.to_numpy(str)
    if len(traits) != 8 or len(times) != 14 or len(ids) != 400:
        raise ValueError("Expected 400 individuals, eight traits and fourteen times")
    predicted = np.full((400, 8, 14), np.nan)
    observed = np.full_like(predicted, np.nan)
    fold_for_sample = np.zeros(400, dtype=int)
    coverage = np.zeros(400, dtype=int)
    sources = ["protocol.json", "sample_ids.csv"]
    for fold in range(1, 6):
        relative = f"main/repeat_{REPEAT:02d}/fold_{fold:02d}/DynaGS/predictions.npz"
        split_file = f"shared/repeat_{REPEAT:02d}/fold_{fold:02d}/split.npz"
        with np.load(SOURCE / relative, allow_pickle=False) as archive:
            index = archive["sample_idx"].astype(int)
            pred, target = archive["predictions"].astype(float), archive["targets"].astype(float)
            if pred.shape != (80, 8, 14) or target.shape != pred.shape or len(np.unique(index)) != 80:
                raise ValueError(f"Unexpected prediction dimensions in {relative}")
            np.testing.assert_array_equal(archive["traits"].astype(str), traits)
            np.testing.assert_array_equal(archive["time"], times)
            np.testing.assert_array_equal(archive["sample_id"].astype(str), ids[index])
        with np.load(SOURCE / split_file, allow_pickle=False) as split:
            np.testing.assert_array_equal(np.sort(index), np.sort(split["test_idx"]))
            if np.intersect1d(index, split["train_idx"]).size or np.intersect1d(index, split["val_idx"]).size:
                raise ValueError("An OOF individual also occurs in the training or validation split")
        if not np.isfinite(pred).all() or not np.isfinite(target).all():
            raise ValueError("Non-finite trajectory values")
        predicted[index], observed[index] = pred, target
        fold_for_sample[index] = fold
        np.add.at(coverage, index, 1)
        sources.extend([relative, split_file])
    if not (coverage == 1).all():
        raise ValueError("Each individual must have exactly one outer-test prediction in this repeat")

    error = predicted - observed
    rmse = np.sqrt(np.mean(error ** 2, axis=2))
    mae = np.mean(np.abs(error), axis=2)
    means = observed.mean(axis=2)
    records = []
    for trait_index, trait in enumerate(traits):
        for index in range(400):
            records.append({"trait": trait, "model": "DynaGS", "repeat": REPEAT,
                            "fold": int(fold_for_sample[index]), "sample_idx": index,
                            "sample_id": ids[index], "observed_mean": means[index, trait_index],
                            "trajectory_rmse": rmse[index, trait_index],
                            "trajectory_mae": mae[index, trait_index], "n_timepoints": len(times)})
    metrics = pd.DataFrame(records)
    samples, trajectories = [], []
    for row, trait in enumerate(traits):
        group = metrics[metrics.trait == trait].sort_values(
            ["trajectory_rmse", "sample_idx"], kind="stable").head(3)
        group = group.sort_values(["observed_mean", "sample_idx"], kind="stable")
        for col, (_, person) in enumerate(group.iterrows()):
            index = int(person.sample_idx)
            panel = chr(ord("A") + row * 3 + col)
            samples.append({"panel": panel, "case_number": col + 1, **person.to_dict()})
            source = f"main/repeat_{REPEAT:02d}/fold_{int(person.fold):02d}/DynaGS/predictions.npz"
            for t_index, time in enumerate(times):
                trajectories.append({"panel": panel, "case_number": col + 1, "trait": trait,
                                     "model": "DynaGS", "repeat": REPEAT, "fold": int(person.fold),
                                     "sample_idx": index, "sample_id": ids[index], "time": time,
                                     "observed": observed[index, row, t_index],
                                     "predicted": predicted[index, row, t_index], "source": source})
    samples, trajectories = pd.DataFrame(samples), pd.DataFrame(trajectories)
    if len(samples) != 24 or len(trajectories) != 336:
        raise ValueError("Expected 24 individual panels and 336 plotted time-point records")
    if samples.duplicated(["trait", "sample_idx"]).any():
        raise ValueError("An individual appears more than once within a trait row")
    return samples, trajectories, metrics, protocol, sources


def draw(samples: pd.DataFrame, data: pd.DataFrame, protocol: dict) -> dict:
    traits = protocol["config"]["phenotype_names"]
    times = np.asarray(protocol["config"]["time_coordinates"], dtype=float)
    limits = {}
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(8, 3, figsize=FIGURE_SIZE, sharex=True, sharey="row")
        fig.subplots_adjust(left=.105, right=.985, top=.925, bottom=.075,
                            wspace=.20, hspace=.45)
        for row, trait in enumerate(traits):
            trait_data = data[data.trait == trait]
            low = float(trait_data[["observed", "predicted"]].to_numpy().min())
            high = float(trait_data[["observed", "predicted"]].to_numpy().max())
            span = max(high - low, 1e-8)
            ylim = (low - .075 * span, high + .075 * span)
            limits[trait] = list(ylim)
            for col in range(3):
                ax = axes[row, col]
                person = samples[(samples.trait == trait) & (samples.case_number == col + 1)].iloc[0]
                frame = trait_data[trait_data.panel == person.panel].sort_values("time")
                np.testing.assert_array_equal(frame.time.to_numpy(), times)
                line, = ax.plot(frame.time, frame.observed, **STYLES["Observed"],
                                linewidth=.9, markersize=2.1, markeredgecolor="white",
                                markeredgewidth=.2, zorder=2)
                np.testing.assert_array_equal(line.get_ydata(), frame.observed.to_numpy())
                line, = ax.plot(frame.time, frame.predicted, **STYLES["DynaGS"],
                                linewidth=1.25, markersize=2.7, markerfacecolor="white",
                                markeredgewidth=.65, markevery=[0, 4, 8, 11, 13], zorder=3)
                np.testing.assert_array_equal(line.get_ydata(), frame.predicted.to_numpy())
                ax.set_ylim(*ylim)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
                formatter = ScalarFormatter(useOffset=False)
                formatter.set_scientific(False)
                ax.yaxis.set_major_formatter(formatter)
                ax.set_xlim(7, 104)
                ax.set_xticks([10, 30, 50, 75, 100])
                ax.tick_params(length=2.1, pad=2, labelsize=6.8, width=.65)
                ax.grid(axis="y", color="#E0E7EC", linewidth=.45)
                ax.spines[["left", "bottom"]].set_color("#9EAFB9")
                ax.spines[["left", "bottom"]].set_linewidth(.65)
                if low < 0 < high:
                    ax.axhline(0, color="#9BAAB4", linestyle=(0, (3, 2)), linewidth=.55, zorder=0)
                ax.set_title(f"{person.panel}  Sample {person.sample_id}", loc="left",
                             fontsize=7.6, color="#193C51", pad=3.5)
                if col == 0:
                    ax.set_ylabel(trait, fontsize=8.2, labelpad=6)

        for col in range(3):
            position = axes[0, col].get_position()
            fig.text(position.x0 + position.width / 2, .958, f"Individual {col + 1}",
                     ha="center", va="center", fontsize=8.1, color="#193C51")
        handles = [Line2D([], [], **STYLES["Observed"], linewidth=.9, markersize=3,
                          markeredgecolor="white", markeredgewidth=.2, label="Observed"),
                   Line2D([], [], **STYLES["DynaGS"], linewidth=1.25, markersize=3,
                          markerfacecolor="white", markeredgewidth=.65, label="DynaGS")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.545, .999),
                   ncol=2, fontsize=8, handlelength=2.7, handletextpad=.65, columnspacing=2.5)
        fig.supxlabel(AXIS_LABELS["time"], x=.545, y=.025, fontsize=8.3)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        validate_figure_axis_labels(fig)
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
    samples, trajectories, metrics, protocol, sources = collect()
    limits = draw(samples, trajectories, protocol)
    DATA.mkdir(parents=True, exist_ok=True)
    for suffix, frame in [("samples", samples), ("trajectories", trajectories), ("individual_metrics", metrics)]:
        frame.to_csv(DATA / f"{STEM}_{suffix}.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_root": str(SOURCE.relative_to(ROOT)),
        "source_files_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in sources},
        "source_values": "Observed and predicted values read directly from the five supplied DynaGS predictions.npz files.",
        "checks": "Validated eight trait names, fourteen time coordinates, sample IDs, outer-test membership, 400 unique OOF individuals and finite original-scale values.",
        "layout": {"rows": 8, "columns": 3},
        "traits": protocol["config"]["phenotype_names"],
        "models": ["DynaGS"], "styles": STYLES,
        "repeat": REPEAT, "outer_folds": 5, "oof_individuals": 400,
        "n_individual_panels": 24, "n_trajectory_records": len(trajectories),
        "model_curves_per_panel": 1,
        "times": protocol["config"]["time_coordinates"],
        "individual_rmse": "sqrt(mean across fourteen times of (predicted-observed)^2), within each individual and trait.",
        "individual_mae": "mean across fourteen times of abs(predicted-observed), within each individual and trait.",
        "line": "Straight segments connecting all fourteen supplied original-scale values for each individual.",
        "markers": "All observed points are displayed; DynaGS markers occur at time indices 0, 4, 8, 11, 13 while its line uses all fourteen values.",
        "axis_limits": limits, "figure_size_inches": list(FIGURE_SIZE), "dpi": 600,
        "figure": str(OUTPUT.relative_to(ROOT)),
        "caption": "图 5-13 DynaGS 在真实玉米八个性状上的个体表型预测轨迹",
        "caption_note": "各行依次对应 IKAW、CIVE、ExB、bn、RGB_G、RGBVI、WI 和 IPCA，每行展示 3 个个体。灰色曲线为观测值，蓝色曲线为 DynaGS 预测值。预测来自第 1 次重复交叉验证中各个体所属外层测试折，每个面板绘制一条折外预测曲线。横坐标采用实际观测时间，同一性状采用相同纵轴范围。",
    }
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(OUTPUT), "n_panels": len(samples), "n_plotted_records": len(trajectories),
                      "n_individual_metrics": len(metrics), "oof_individuals": 400}, ensure_ascii=False))


if __name__ == "__main__":
    main()
