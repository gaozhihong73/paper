"""Plot the current validated CSV data for eight real maize traits."""

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
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from dynags_figures.style import thesis_style

DATA = PROJECT / "data"
STEM = "figure_5_11_real_trait_performance"
OUTPUT = ROOT / "07_论文图表/5-11_真实玉米各性状的整体预测性能比较.png"
MODELS = ["Mean", "Ridge-A+D", "GBLUP", "XGBoost", "Cropformer", "DynaGS"]
BASELINES = MODELS[:-1]
INPUTS = {suffix: DATA / f"{STEM}_{suffix}.csv" for suffix in ("runs", "pairs", "summary")}


def collect() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    metadata = json.loads((DATA / f"{STEM}.json").read_text(encoding="utf-8"))
    traits = metadata["traits"]
    repeats = range(1, 6)
    if len(traits) != 8 or len(repeats) != 5:
        raise ValueError("Expected eight traits and five independent repeats")

    runs = pd.read_csv(INPUTS["runs"], dtype={"time": "string"})
    pairs = pd.read_csv(INPUTS["pairs"], dtype={"time": "string"})
    cells = pd.read_csv(INPUTS["summary"])

    if len(runs) != 240 or len(pairs) != 200 or len(cells) != 88:
        raise ValueError("Unexpected number of CSV records")
    if runs.duplicated(["repeat", "trait", "model"]).any():
        raise ValueError("Duplicate absolute repeat records")
    if pairs.duplicated(["repeat", "trait", "reference"]).any():
        raise ValueError("Duplicate paired repeat records")
    if cells.duplicated(["panel", "model", "trait"]).any():
        raise ValueError("Duplicate summary records")
    if not ((runs.fold == 0).all() and (runs.n_test == 400).all()
            and (runs.valid_count == 14).all() and runs.valid.all()
            and (runs.suite == "main").all() and (runs.fraction == 1.).all()
            and (runs.time == "all").all() and (runs.metric == "pcc").all()):
        raise ValueError("Expected complete, repeat-level pooled OOF PCC records")
    if not ((pairs.comparison == "model").all() and (pairs.time == "all").all()
            and (pairs.metric == "pcc").all()):
        raise ValueError("Unexpected paired comparison records")
    if not np.isfinite(runs.value).all() or not np.isfinite(pairs.value).all() or not np.isfinite(cells["mean"]).all():
        raise ValueError("Non-finite PCC values require explicit missing-value handling")

    wide = runs.pivot(index=["repeat", "trait"], columns="model", values="value")
    pair_wide = pairs.pivot(index=["repeat", "trait"], columns="reference", values="value")
    expected = pd.MultiIndex.from_product([repeats, traits], names=["repeat", "trait"])
    wide = wide.loc[expected, MODELS]
    pair_wide = pair_wide.loc[expected, BASELINES]
    for baseline in BASELINES:
        np.testing.assert_allclose(pair_wide[baseline], wide.DynaGS - wide[baseline],
                                   rtol=0, atol=1e-12)
    expected_improves = pairs.value > 0
    if not np.array_equal(pairs.improves.to_numpy(), expected_improves.to_numpy()):
        raise ValueError("Paired improvement flags do not match differences")

    summary = cells.set_index(["panel", "model", "trait"])
    for panel, models, source, field in [
        ("A", MODELS, runs, "model"),
        ("B", BASELINES, pairs, "reference"),
    ]:
        for model in models:
            for trait in traits:
                values = source[(source[field] == model) & (source.trait == trait)]
                values = values.set_index("repeat").loc[list(repeats), "value"]
                row = summary.loc[(panel, model, trait)]
                np.testing.assert_allclose([row["mean"], row["sample_sd"]],
                                           [values.mean(), values.std(ddof=1)],
                                           rtol=0, atol=1e-12)
                if int(row.n_repeats) != len(repeats):
                    raise ValueError("Summary repeat count does not match its source")
                expected_metric = "pcc" if panel == "A" else "delta_pcc"
                if row.metric != expected_metric:
                    raise ValueError("Summary metric does not match its panel")
                if panel == "B" and int((values > 0).sum()) != int(row.n_improved):
                    raise ValueError("Paired improvement count does not match the repeat records")
    return runs, pairs, cells, metadata


def draw(cells: pd.DataFrame, traits: list[str]) -> dict:
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig = plt.figure(figsize=(6.3, 5.1))
        grid = fig.add_gridspec(2, 2, width_ratios=[1, .035], height_ratios=[6, 5],
                               left=.18, right=.89, top=.93, bottom=.085,
                               hspace=.43, wspace=.07)
        image_checks = {}
        number_annotations = []
        for i, (panel, models, title, cmap, norm, ticks, label) in enumerate([
            ("A", MODELS, "A  Mean PCC across repeats", "Blues", Normalize(-.1, .8),
             [-.1, 0, .2, .4, .6, .8], "PCC"),
            ("B", BASELINES, "B  Paired PCC difference", "RdBu", TwoSlopeNorm(0, -.8, .8),
             [-.8, -.4, 0, .4, .8], r"$\Delta\mathrm{PCC}$"),
        ]):
            ax = fig.add_subplot(grid[i, 0])
            cax = fig.add_subplot(grid[i, 1])
            part = cells[cells.panel == panel].pivot(index="model", columns="trait", values="mean")
            values = part.loc[models, traits].to_numpy(float)
            im = ax.imshow(values, cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
            np.testing.assert_array_equal(np.asarray(im.get_array()), values)
            ax.set_xticks(range(len(traits)), labels=traits)
            ax.set_yticks(range(len(models)), labels=models)
            ax.tick_params(axis="x", length=0, pad=6, labelsize=8)
            ax.tick_params(axis="y", length=0, pad=7, labelsize=8)
            ax.set_xticks(np.arange(-.5, len(traits), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(models), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=.65)
            ax.tick_params(which="minor", length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(title, fontsize=8.6, color="#193C51", loc="left", pad=10)
            for row, col in np.ndindex(values.shape):
                rgb = np.asarray(im.cmap(im.norm(values[row, col])))[:3]
                linear_rgb = np.where(rgb <= .04045, rgb / 12.92,
                                      ((rgb + .055) / 1.055) ** 2.4)
                luminance = float(linear_rgb @ np.array([.2126, .7152, .0722]))
                color = "white" if 1.05 / (luminance + .05) > (luminance + .05) / .05 else "black"
                value_label = f"{values[row, col]:.3f}".replace("-", "\N{MINUS SIGN}")
                text = ax.text(col, row, value_label,
                               ha="center", va="center", color=color, fontsize=7.2,
                               fontweight="normal", zorder=4)
                number_annotations.append((text, ax, row, col))
            if panel == "B":
                negative_cells = np.argwhere(values < 0)
                for row, col in negative_cells:
                    ax.add_patch(Rectangle((col - .445, row - .435), .89, .87,
                                           facecolor="none", edgecolor="#626F78",
                                           linewidth=.75, linestyle=(0, (3, 2))))
                if len(negative_cells):
                    handle = Patch(facecolor="none", edgecolor="#626F78", linewidth=.75,
                                   linestyle=(0, (3, 2)), label=r"$\Delta\mathrm{PCC}<0$")
                    ax.legend(handles=[handle], loc="lower right", bbox_to_anchor=(1, 1.025),
                              borderaxespad=0, fontsize=7.2, handlelength=1.6, handletextpad=.5)
            cb = fig.colorbar(im, cax=cax, ticks=ticks)
            cb.outline.set_visible(False)
            cb.ax.tick_params(length=2.3, width=.6, pad=3, labelsize=7.4)
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            cb.set_label(label, fontsize=8.3, labelpad=6)
            cb.ax.set_ylim(norm.vmin, norm.vmax)
            image_checks[panel] = {"shape": list(values.shape), "min": float(values.min()),
                                    "max": float(values.max()), "negative_cells": int((values < 0).sum())}
            image_checks[panel]["value_labels"] = int(values.size)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text, ax, row, col in number_annotations:
            box = text.get_window_extent(renderer)
            left, bottom = ax.transData.transform((col - .4, row + .35))
            right, top = ax.transData.transform((col + .4, row - .35))
            if box.x0 < left or box.x1 > right or box.y0 < bottom or box.y1 > top:
                raise ValueError(f"Value annotation extends outside its cell: {text.get_text()}")
        for item in fig.findobj(matplotlib.text.Text):
            if not item.get_visible() or not item.get_text():
                continue
            box = item.get_window_extent(renderer)
            if box.width == 0 or box.height == 0:
                continue
            if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or box.y1 > fig.bbox.y1 + 1:
                raise ValueError(f"Text extends outside the publication canvas: {item.get_text()}")
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)
    return image_checks


def main() -> None:
    runs, pairs, cells, metadata = collect()
    traits = metadata["traits"]
    image_checks = draw(cells, traits)
    metadata.pop("source_root", None)
    original_hashes = metadata.pop("source_files_sha256", None)
    if original_hashes is not None:
        metadata["upstream_original_source_files_sha256"] = original_hashes
    metadata["plot_input_files_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in INPUTS.values()
    }
    metadata["checks"] = (
        "CSV row counts, pairing, improvement flags, repeat means, sample SD, "
        "repeat counts and improvement counts verified."
    )
    metadata["display"]["cell_checks"] = image_checks
    metadata["display"]["A"] = (
        "Six models by eight traits; mean PCC labels to three decimal places."
    )
    metadata["display"]["B"] = (
        "Five baselines by eight traits; mean paired delta PCC labels to three decimal places; "
        "dashed borders identify negative differences."
    )
    metadata["caption_note"] = (
        "A：6 种模型在 8 个性状上的 PCC 均值；"
        "B：DynaGS 相对各对照模型的配对 PCC 差值均值。两个面板的数值均保留三位小数。"
        "每次重复先合并 5 个外层测试折的折外预测，在各时间点计算 PCC 并取有效时间点均值，"
        "再对 5 次重复取平均。ΔPCC 为同一重复、同一性状上 DynaGS 的 PCC 减去对照模型的 PCC，"
        "正值表示 DynaGS 的 PCC 更高；虚线框表示均值小于 0。两部分分别使用各自的色标。"
    )
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"validated_runs": len(runs), "validated_pairs": len(pairs),
                      "plotted_cells": len(cells), "figure": str(OUTPUT),
                      "image_checks": image_checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
