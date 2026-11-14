"""Plot SNP-set overlap and selection counts over the 25 main outer folds."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from dynags_figures.style import thesis_style
from dynags_figures.labels import validate_figure_axis_labels

SOURCE = ROOT / "05_实验结果/5.5/outputs/tfpp/maize"
DATA = PROJECT / "data"
STEM = "figure_5_16_snp_screening_stability"
OUTPUT = ROOT / "07_论文图表/5-16_真实玉米数据中SNP筛选结果的稳定性.png"
BLUE = "#0072B2"
INK = "#193C51"
SELF_COLOR = "#E8EEF2"
FILES = ["protocol.json", "analysis/snp_jaccard.csv", "analysis/snp_frequency.csv",
         "analysis/screening_counts.csv"]


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    candidate_count = int(protocol["shape"][1])
    expected_size = int(protocol["config"]["screen_final_k"])
    supplied_matrix = pd.read_csv(SOURCE / "analysis/snp_jaccard.csv", index_col=0)
    supplied_frequency = pd.read_csv(SOURCE / "analysis/snp_frequency.csv")
    counts = pd.read_csv(SOURCE / "analysis/screening_counts.csv")
    if len(counts) != 25 or counts.duplicated(["repeat", "fold"]).any():
        raise ValueError("Expected 25 unique main-fold screening records")
    sets, labels, paths, fold_rows = [], [], [], []
    for repeat in range(1, 6):
        for fold in range(1, 6):
            relative = f"shared/repeat_{repeat:02d}/fold_{fold:02d}/selected.npy"
            values = np.load(SOURCE / relative, allow_pickle=False)
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError(f"Expected one-dimensional integer SNP indices: {relative}")
            if len(values) != expected_size or len(np.unique(values)) != expected_size:
                raise ValueError("Unexpected number of unique selected SNPs")
            if values.min() < 0 or values.max() >= candidate_count:
                raise ValueError("SNP index lies outside the candidate matrix")
            row = counts[(counts["repeat"] == repeat) & (counts.fold == fold)]
            if len(row) != 1 or int(row.final_snps.iloc[0]) != len(values):
                raise ValueError("Selected SNP count differs from the supplied screening log")
            if int(row.raw_snps.iloc[0]) != candidate_count:
                raise ValueError("Candidate SNP count differs from the frozen protocol")
            label = f"r{repeat}_f{fold}"
            sets.append(set(values.tolist()))
            labels.append(label)
            paths.append(relative)
            fold_rows.append({"fold_label": label, "repeat": repeat, "fold": fold,
                              "selected_snps": len(values), "candidate_snps": candidate_count,
                              "source": relative})
    if supplied_matrix.index.tolist() != labels or supplied_matrix.columns.tolist() != labels:
        raise ValueError("Matrix labels differ from the repeat-major, fold-minor order")

    matrix = np.eye(25)
    pair_rows = []
    for i in range(25):
        for j in range(i + 1, 25):
            intersection = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            value = intersection / union
            matrix[i, j] = matrix[j, i] = value
            pair_rows.append({"fold_a": labels[i], "fold_b": labels[j],
                              "repeat_a": i // 5 + 1, "repeat_b": j // 5 + 1,
                              "comparison": "Within repeat" if i // 5 == j // 5 else "Between repeats",
                              "intersection_snps": intersection, "union_snps": union,
                              "jaccard_similarity": value})
    np.testing.assert_allclose(matrix, supplied_matrix.to_numpy(float), rtol=0, atol=1e-12)
    matrix_frame = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_frame.index.name = "fold_label"
    pairs = pd.DataFrame(pair_rows)

    frequency = Counter(value for values in sets for value in values)
    frequency_frame = pd.DataFrame([
        {"snp_original_index": index, "selected_count": frequency[index], "n_folds": 25}
        for index in sorted(frequency)
    ])
    if supplied_frequency.duplicated("snp_original_index").any():
        raise ValueError("Duplicate SNP indices in the supplied frequency table")
    actual = supplied_frequency.sort_values("snp_original_index").reset_index(drop=True)
    pd.testing.assert_frame_equal(frequency_frame, actual, check_dtype=False)
    histogram = np.bincount(frequency_frame.selected_count, minlength=26)
    histogram[0] = candidate_count - len(frequency_frame)
    if histogram.sum() != candidate_count or np.dot(np.arange(26), histogram) != 25 * expected_size:
        raise ValueError("Selection-count distribution does not match candidate and selected totals")
    distribution = pd.DataFrame({"selection_count": np.arange(26), "n_snps": histogram})
    distribution["fraction_of_candidates"] = distribution.n_snps / candidate_count
    if len(pairs) != 300 or (pairs.comparison == "Within repeat").sum() != 50:
        raise ValueError("Unexpected number of pairwise fold comparisons")
    return matrix_frame, pairs, frequency_frame, distribution, pd.DataFrame(fold_rows), paths


def draw(matrix: pd.DataFrame, pairs: pd.DataFrame, distribution: pd.DataFrame) -> dict:
    display_labels = [label.replace("r", "R").replace("_f", "F") for label in matrix.index]
    self_mask = np.eye(25, dtype=bool)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad(SELF_COLOR)
    color_limits = (.25, .40)
    if pairs.jaccard_similarity.min() < color_limits[0] or pairs.jaccard_similarity.max() > color_limits[1]:
        raise ValueError("Color scale would truncate off-diagonal similarity values")
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig = plt.figure(figsize=(6.3, 4.3))
        heat = fig.add_axes((.092, .19, .445, .652))
        cax = fig.add_axes((.55, .19, .017, .652))
        pair_ax = fig.add_axes((.718, .59, .268, .255))
        freq_ax = fig.add_axes((.718, .19, .268, .26))

        im = heat.imshow(np.ma.array(matrix.to_numpy(), mask=self_mask), cmap=cmap,
                         norm=Normalize(*color_limits), interpolation="nearest", aspect="auto")
        heat.set_xticks(range(25), labels=display_labels, rotation=90)
        heat.set_yticks(range(25), labels=display_labels)
        heat.tick_params(length=0, pad=3, labelsize=6.6)
        heat.set_xticks(np.arange(-.5, 25, 1), minor=True)
        heat.set_yticks(np.arange(-.5, 25, 1), minor=True)
        heat.grid(which="minor", color="white", linewidth=.22)
        heat.tick_params(which="minor", length=0)
        for boundary in [4.5, 9.5, 14.5, 19.5]:
            heat.axvline(boundary, color="white", linewidth=1.1)
            heat.axhline(boundary, color="white", linewidth=1.1)
        for spine in heat.spines.values():
            spine.set_visible(False)
        heat.set_xlabel("Outer training fold", fontsize=7.7, labelpad=6)
        heat.set_ylabel("Outer training fold", fontsize=7.7, labelpad=6)
        heat.set_title("A  SNP set similarity", loc="left", fontsize=8.3, color=INK, pad=8)
        cb = fig.colorbar(im, cax=cax, ticks=[.25, .30, .35, .40])
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=2.3, pad=3, labelsize=7)
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cb.set_label("Jaccard similarity", fontsize=7.7, labelpad=6)

        categories = ["Within repeat", "Between repeats"]
        groups = [pairs.loc[pairs.comparison == category, "jaccard_similarity"].to_numpy()
                  for category in categories]
        rng = np.random.default_rng(73)
        for position, group in enumerate(groups, start=1):
            pair_ax.scatter(position + rng.uniform(-.14, .14, len(group)), group,
                            s=3.8, color="#8295A0", alpha=.38, edgecolors="none", zorder=2)
        pair_ax.boxplot(groups, positions=[1, 2], widths=.43, patch_artist=True,
                        showfliers=False, whis=1.5,
                        boxprops={"facecolor": "#DDEAF2", "edgecolor": BLUE, "linewidth": .9},
                        medianprops={"color": "#193C51", "linewidth": 1.2},
                        whiskerprops={"color": BLUE, "linewidth": .85},
                        capprops={"color": BLUE, "linewidth": .85}, zorder=3)
        pair_ax.set_xticks([1, 2], labels=["Within repeat\n50 pairs", "Between repeats\n250 pairs"])
        pair_ax.set_xlim(.5, 2.5)
        pair_ax.set_ylim(.24, .40)
        pair_ax.set_yticks([.25, .30, .35, .40])
        pair_ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        pair_ax.set_ylabel("Jaccard similarity", fontsize=7.5, labelpad=6)
        pair_ax.set_title("B  Pairwise Jaccard", loc="left", fontsize=8.1, color=INK, pad=8)
        pair_ax.tick_params(axis="x", length=0, pad=4, labelsize=7)

        shown = distribution[distribution.selection_count >= 1]
        freq_ax.bar(shown.selection_count, shown.n_snps, width=.84, color=BLUE,
                    edgecolor="white", linewidth=.25, zorder=3)
        freq_ax.set_xlim(.25, 25.75)
        freq_ax.set_ylim(0, float(shown.n_snps.max()) * 1.16)
        freq_ax.set_xticks([1, 5, 10, 15, 20, 25])
        freq_ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        freq_ax.set_xlabel("Folds selecting a SNP", fontsize=7.5, labelpad=6)
        freq_ax.set_ylabel("Number of SNPs", fontsize=7.5, labelpad=6)
        freq_ax.set_title("C  SNP selection frequency", loc="left", fontsize=8.1, color=INK, pad=8)
        count25 = int(distribution.loc[distribution.selection_count == 25, "n_snps"].iloc[0])
        freq_ax.text(.98, .95, f"25 folds: {count25}", transform=freq_ax.transAxes,
                     ha="right", va="top", fontsize=6.9, color="#455560")
        for ax in [pair_ax, freq_ax]:
            ax.tick_params(axis="y", length=2.2, pad=2, labelsize=7, width=.65)
            ax.spines[["left", "bottom"]].set_color("#9EAFB9")
            ax.spines[["left", "bottom"]].set_linewidth(.65)
            ax.grid(axis="y", color="#E0E7EC", linewidth=.45)
        freq_ax.tick_params(axis="x", length=2.2, pad=2.5, labelsize=7, width=.65)
        fig.legend(handles=[Patch(facecolor=SELF_COLOR, edgecolor="none", label="Self-comparison")],
                   loc="lower center", bbox_to_anchor=(.315, .015), fontsize=7,
                   handlelength=1.3, handletextpad=.5)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        validate_figure_axis_labels(fig)
        if abs(heat.bbox.height - cax.bbox.height) > .5:
            raise ValueError("Colorbar and heatmap heights differ")
        text_items = fig.legends[0].get_texts()
        for ax in fig.axes:
            text_items += [ax._left_title, ax.xaxis.label, ax.yaxis.label]
            text_items += ax.texts
            xlo, xhi = sorted(ax.get_xlim()); ylo, yhi = sorted(ax.get_ylim())
            text_items += [label for value, label in zip(ax.get_xticks(), ax.get_xticklabels()) if xlo <= value <= xhi]
            text_items += [label for value, label in zip(ax.get_yticks(), ax.get_yticklabels()) if ylo <= value <= yhi]
        for item in text_items:
            if not item.get_visible() or not item.get_text():
                continue
            box = item.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
                raise ValueError(f"Text extends outside the publication canvas: {item.get_text()}")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)
    return {"jaccard_color_limits": list(color_limits), "pairwise_y_limits": [.24, .40]}


def main() -> None:
    matrix, pairs, frequency, distribution, folds, paths = collect()
    limits = draw(matrix, pairs, distribution)
    DATA.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(DATA / f"{STEM}_jaccard.csv", encoding="utf-8-sig")
    for suffix, frame in [("pairs", pairs), ("frequency", frequency),
                          ("frequency_distribution", distribution), ("folds", folds)]:
        frame.to_csv(DATA / f"{STEM}_{suffix}.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_root": str(SOURCE.relative_to(ROOT)),
        "source_files_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in FILES + paths},
        "checks": "All 25 supplied selected.npy sets recomputed and matched against the supplied Jaccard, frequency and screening-count tables.",
        "fold_order": folds.fold_label.tolist(), "candidate_snps": int(distribution.n_snps.sum()),
        "snps_per_fold": 5000, "n_folds": 25, "n_pairs": len(pairs),
        "jaccard": "Intersection size divided by union size for each pair of selected SNP sets.",
        "pair_summary": pairs.groupby("comparison").jaccard_similarity.agg(["count", "mean", "median", "min", "max"]).to_dict("index"),
        "selected_union_snps": len(frequency),
        "never_selected_snps": int(distribution.loc[distribution.selection_count == 0, "n_snps"].iloc[0]),
        "selected_in_all_25": int(distribution.loc[distribution.selection_count == 25, "n_snps"].iloc[0]),
        "panel_A": "Repeat-major fold order; self-comparisons masked in light grey; no numeric cell annotations.",
        "panel_B": "All 300 unordered non-self fold pairs, separated into 50 within-repeat and 250 between-repeat pairs; box shows quartiles and median, whiskers use 1.5 IQR; all points displayed.",
        "panel_C": "SNP counts by the number of selecting folds, among SNPs appearing at least once; zero-count candidates retained in the saved distribution CSV.",
        "jitter": "Horizontal category jitter in panel B, seed 73; similarity values are unchanged.",
        "axis_limits": limits, "figure_size_inches": [6.3, 4.3], "dpi": 600,
        "figure": str(OUTPUT.relative_to(ROOT)),
        "caption": "图 5-16 真实玉米数据中 SNP 筛选结果的稳定性",
        "caption_note": "A：5 次重复、每次 5 个外层训练折的 SNP 集合 Jaccard 相似性，R 和 F 分别表示重复与折编号；浅灰色对角线为自身比较，白色分隔线区分重复。B：同一重复内 50 对训练折与不同重复间 250 对训练折的相似性分布；箱体为四分位区间，中线为中位数，须线延伸至 1.5 倍四分位距内的实际观测值，散点为各折对。C：至少一次入选的 17,553 个 SNP 在 25 个训练折中的入选次数分布，其中 121 个 SNP 在全部 25 个折中入选。各折最终位点数均为 5,000。",
    }
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": str(OUTPUT), "folds": len(folds), "pairs": len(pairs),
                      "union_snps": len(frequency), "all_25": metadata["selected_in_all_25"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
