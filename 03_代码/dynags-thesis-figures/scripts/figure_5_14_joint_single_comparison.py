"""Plot within-repeat joint-versus-single-trait DynaGS differences."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import Formatter, FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from dynags_figures.style import thesis_style

SOURCE = ROOT / "05_实验结果/5.5/outputs/tfpp/maize"
DATA = PROJECT / "data"
STEM = "figure_5_14_joint_single_comparison"
OUTPUT = ROOT / "07_论文图表/5-14_单性状与多性状联合训练的配对预测性能差异.png"
METRICS = ["pcc", "r2", "rmse"]
FILES = ["protocol.json", "analysis/summary_protocol.json", "analysis/repeat_metrics.csv",
         "analysis/summary.csv", "analysis/paired_differences.csv", "analysis/paired_summary.csv"]


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    traits = protocol["config"]["phenotype_names"]
    repeats = list(range(1, len(protocol["protocol"]["seeds"]) + 1))
    if len(traits) != 8 or len(repeats) != 5:
        raise ValueError("Expected eight traits and five paired repeats")

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(SOURCE / "analysis" / name, dtype={"time": "string"})

    runs = read("repeat_metrics.csv")
    runs = runs[(runs.model == "DynaGS") & runs.suite.isin(["main", "single"])
                & (runs.time == "all") & (runs.fraction == 1.) & runs.metric.isin(METRICS)].copy()
    pairs = read("paired_differences.csv")
    pairs = pairs[(pairs.comparison == "joint_vs_single") & (pairs.time == "all")
                  & pairs.metric.isin(METRICS)].copy()
    absolute = read("summary.csv")
    absolute = absolute[(absolute.model == "DynaGS") & absolute.suite.isin(["main", "single"])
                        & (absolute.time == "all") & (absolute.fraction == 1.)
                        & absolute.metric.isin(METRICS)].set_index(["suite", "trait", "metric"])
    paired = read("paired_summary.csv")
    paired = paired[(paired.comparison == "joint_vs_single") & (paired.time == "all")
                    & paired.metric.isin(METRICS)].set_index(["trait", "metric"])
    if len(runs) != 240 or len(pairs) != 120 or len(absolute) != 48 or len(paired) != 24:
        raise ValueError("Unexpected number of joint-versus-single records")
    if runs.duplicated(["suite", "trait", "metric", "repeat"]).any():
        raise ValueError("Duplicate absolute repeat records")
    if pairs.duplicated(["trait", "metric", "repeat"]).any():
        raise ValueError("Duplicate paired repeat records")
    if not absolute.index.is_unique or not paired.index.is_unique:
        raise ValueError("Duplicate supplied summary records")
    if not ((runs.fold == 0).all() and (runs.n_test == 400).all() and runs.valid.all()):
        raise ValueError("Expected complete repeat-level OOF results on 400 samples")
    if not (pairs.reference == "single").all():
        raise ValueError("Unexpected paired reference condition")
    if not np.isfinite(runs.value).all() or not np.isfinite(pairs.value).all():
        raise ValueError("Missing metrics need explicit handling")

    joint_tasks = [task for task in protocol["tasks"]
                   if task["suite"] == "main" and task["model"] == "DynaGS"]
    single_tasks = [task for task in protocol["tasks"]
                    if task["suite"] == "single" and task["model"] == "DynaGS"]
    if len(joint_tasks) != 25 or len(single_tasks) != 200:
        raise ValueError("Expected 25 joint-trait tasks and 200 single-trait tasks")
    shared = {(task["repeat"], task["fold"]): task["shared"] for task in joint_tasks}
    for task in single_tasks:
        if task["shared"] != shared[(task["repeat"], task["fold"])]:
            raise ValueError("Joint and single tasks refer to different shared split/screening inputs")

    rows = []
    for trait in traits:
        for metric in METRICS:
            condition = runs[(runs.trait == trait) & (runs.metric == metric)]
            values = condition.pivot(index="repeat", columns="suite", values="value").loc[repeats]
            saved = pairs[(pairs.trait == trait) & (pairs.metric == metric)].set_index("repeat").loc[repeats]
            difference = values["main"] - values["single"]
            if metric == "rmse":
                difference = -difference
            np.testing.assert_allclose(saved.value, difference, rtol=0, atol=1e-12)
            np.testing.assert_array_equal(saved.improves.to_numpy(), difference.to_numpy() > 0)
            source = paired.loc[(trait, metric)]
            mean, sd = float(saved.value.mean()), float(saved.value.std(ddof=1))
            np.testing.assert_allclose([source["mean"], source["std"]], [mean, sd], rtol=0, atol=1e-12)
            raw = json.loads(source.raw_repeats)
            np.testing.assert_allclose([raw[str(i)] for i in repeats], saved.value, rtol=0, atol=1e-12)
            if int(source.n_valid) != 5 or int(source.n_improved) != int((saved.value > 0).sum()):
                raise ValueError("Supplied paired counts differ from repeat records")
            record = {"trait": trait, "metric": metric, "n_pairs": 5,
                      "mean_difference": mean, "sd_difference": sd,
                      "sem_difference": sd / np.sqrt(5), "df": 4,
                      "t_critical_975": float(student_t.ppf(.975, 4)),
                      "n_improved": int((saved.value > 0).sum())}
            half_width = record["t_critical_975"] * record["sem_difference"]
            record["ci95_lower"], record["ci95_upper"] = mean - half_width, mean + half_width
            for suite, label in [("main", "joint"), ("single", "single")]:
                row = absolute.loc[(suite, trait, metric)]
                np.testing.assert_allclose([row["mean"], row["std"]],
                                           [values[suite].mean(), values[suite].std(ddof=1)],
                                           rtol=0, atol=1e-12)
                raw = json.loads(row.raw_repeats)
                np.testing.assert_allclose([raw[str(i)] for i in repeats], values[suite], rtol=0, atol=1e-12)
                if int(row.n_valid) != 5:
                    raise ValueError("Unexpected absolute summary repeat count")
                record[f"{label}_mean"], record[f"{label}_sd"] = row["mean"], row["std"]
            rows.append(record)
    return runs, pairs, pd.DataFrame(rows), protocol


def draw(pairs: pd.DataFrame, summary: pd.DataFrame, traits: list[str]) -> dict:
    limits = {}
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 4.2), sharey=True)
        interval_handle = None
        for j, metric in enumerate(["pcc", "r2"]):
            ax = axes[j]
            part = summary[summary.metric == metric].set_index("trait").loc[traits]
            raw = pairs[pairs.metric == metric]
            lo = min(0., float(part.ci95_lower.min()), float(raw.value.min()))
            hi = max(0., float(part.ci95_upper.max()), float(raw.value.max()))
            pad = .10 * (hi - lo)
            xlim = (lo - pad, hi + pad)
            limits[metric] = list(xlim)
            for index, trait in enumerate(traits):
                if index % 2 == 0:
                    ax.axhspan(index - .5, index + .5, color="#F2F6F8", linewidth=0, zorder=0)
                row = part.loc[trait]
                values = raw[raw.trait == trait].sort_values("repeat").value.to_numpy(float)
                point_y = index + .16 + np.linspace(-.08, .08, 5)
                ax.scatter(values, point_y, s=10, color="#8295A0", edgecolors="white",
                           linewidths=.3, alpha=.90, zorder=3)
                handle = ax.errorbar(row.mean_difference, index - .15,
                                     xerr=np.array([[row.mean_difference - row.ci95_lower],
                                                    [row.ci95_upper - row.mean_difference]]),
                                     fmt="D", color="#0072B2", markersize=4.3,
                                     markeredgecolor="white", markeredgewidth=.5,
                                     elinewidth=1.05, capsize=2.5, capthick=.8, zorder=4,
                                     label="Mean (95% CI)" if j == 0 and index == 0 else None)
                if interval_handle is None:
                    interval_handle = handle
            ax.axvline(0, color="#8195A0", linestyle=(0, (3, 2)), linewidth=.9, zorder=2)
            ax.set_xlim(*xlim)
            ax.set_ylim(len(traits) - .5, -.55)
            ax.set_yticks(range(len(traits)), labels=traits)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
            precision = 2 if metric == "pcc" else 3
            ax.xaxis.set_major_formatter(FuncFormatter(
                lambda value, _, digits=precision: Formatter.fix_minus(f"{value:.{digits}f}")))
            ax.tick_params(axis="y", length=0, pad=7, labelsize=8)
            ax.tick_params(axis="x", length=2.5, pad=3, labelsize=7.1)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#9EAFB9")
            ax.grid(axis="x", color="#E0E7EC", linewidth=.5)
            ax.set_title("A  PCC difference" if metric == "pcc" else r"B  Time-adjusted $R^2$ difference",
                         loc="left", fontsize=8.3, color="#193C51", pad=10)
            ax.set_xlabel(r"$\Delta\mathrm{PCC}$" if metric == "pcc" else r"$\Delta R^2_{\mathrm{time}}$",
                          fontsize=8.4, labelpad=7)
        axes[1].tick_params(axis="y", labelleft=False)
        repeat_handle = Line2D([], [], marker="o", linestyle="none", color="#8295A0",
                               markersize=3.5, markeredgecolor="white", markeredgewidth=.3,
                               label="Repeat")
        fig.legend(handles=[repeat_handle, interval_handle], labels=["Repeat", "Mean (95% CI)"],
                   loc="upper center", bbox_to_anchor=(.55, .989), ncol=2,
                   fontsize=7.5, handlelength=2.4, handletextpad=.65, columnspacing=2.)
        fig.text(.55, .055, "Positive values favor joint training", ha="center", va="center",
                 fontsize=7.8, color="#455560")
        fig.subplots_adjust(left=.14, right=.98, top=.855, bottom=.24, wspace=.24)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        text_items = list(fig.texts) + fig.legends[0].get_texts()
        for ax in axes:
            text_items += [ax._left_title, ax.xaxis.label] + ax.get_yticklabels()
            text_items += [label for value, label in zip(ax.get_xticks(), ax.get_xticklabels())
                           if ax.get_xlim()[0] <= value <= ax.get_xlim()[1]]
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
    runs, pairs, summary, protocol = collect()
    limits = draw(pairs, summary, protocol["config"]["phenotype_names"])
    DATA.mkdir(parents=True, exist_ok=True)
    for suffix, frame in [("runs", runs), ("pairs", pairs), ("summary", summary)]:
        frame.to_csv(DATA / f"{STEM}_{suffix}.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_root": str(SOURCE.relative_to(ROOT)),
        "source_files_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
                                for name in FILES},
        "source_filter": {"model": "DynaGS", "suite": ["main", "single"], "time": "all",
                          "paired_comparison": "joint_vs_single", "paired_reference": "single"},
        "source_values": "Provided repeat-level OOF metrics and paired differences; no metrics recomputed from predictions.",
        "checks": "Verified within-repeat differences, supplied means/SD/raw_repeats/counts and shared split/screening task references for 25 joint plus 200 single-trait tasks.",
        "directions": {"pcc": "PCC_joint minus PCC_single", "r2": "R2_time_joint minus R2_time_single",
                       "rmse": "RMSE_single minus RMSE_joint on each trait's original scale"},
        "interval": "Two-sided 95% Student-t interval for the mean of five paired repeat differences: mean +/- t(0.975, df=4) * sample_SD / sqrt(5); ddof=1.",
        "points": "Five provided repeat differences shown as small grey points; blue diamonds and error bars show the paired mean and its 95% CI.",
        "vertical_offsets": "Within-row offsets separate repeat points from the mean interval; vertical positions carry no numerical meaning.",
        "displayed_metrics": ["pcc", "r2"], "saved_metrics": METRICS,
        "rmse": "Preserved separately for all eight traits without averaging different trait scales.",
        "traits": protocol["config"]["phenotype_names"],
        "n_pairs_per_trait": 5, "n_oof_samples_per_repeat": 400,
        "n_absolute_records": len(runs), "n_pair_records": len(pairs),
        "axis_limits": limits, "figure_size_inches": [6.3, 4.2], "dpi": 600,
        "figure": str(OUTPUT.relative_to(ROOT)),
        "caption": "图 5-14 单性状与多性状联合训练的配对预测性能差异",
        "caption_note": "A：联合训练相对单性状训练的 PCC 差值；B：时间校正 R² 差值。灰色点表示 5 次重复的配对差值，蓝色菱形及误差线表示配对差值均值和基于 5 个重复的双侧 95% Student-t 置信区间（自由度为 4）。各项差值均为联合训练减单性状训练，正值表示联合训练对应指标更高。每次重复使用合并 5 个外层测试折后的折外预测计算指标。",
    }
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(OUTPUT), "absolute_records": len(runs), "pair_records": len(pairs),
                      "summary_records": len(summary), "axis_limits": limits}, ensure_ascii=False))


if __name__ == "__main__":
    main()
