"""Audit all existing figures and refresh approved labels using saved plot data."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
REPORT = ROOT / "08_阶段产物/图件坐标轴命名核查"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
from dynags_figures.labels import AXIS_LABELS, figure_axis_labels, validate_figure_axis_labels


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(key: str, module):
    def csv(name, **kwargs):
        return pd.read_csv(DATA / name, **kwargs)

    name = module.__name__
    protocol = lambda: json.loads((ROOT / "05_实验结果/5.5/outputs/tfpp/maize/protocol.json").read_text(encoding="utf-8"))
    if key == "3_1":
        return (module.inspect_inputs(),)
    if key == "3_2":
        return (csv("figure_3_2_qc_counts.csv").to_dict("records"),)
    if key == "3_3":
        return (csv("figure_3_3_snp_density_1mb.csv").to_dict("records"),)
    if key == "3_4":
        return (csv("figure_3_4_trait_trajectory_summary.csv").to_dict("records"),)
    if key == "3_5":
        frame = csv("figure_3_5_standardized_mean_trajectories.csv")
        matrix = frame.pivot(index="trait", columns="time_coordinate", values="z_score")
        return (matrix.loc[module.TRAITS, module.TIMES].to_numpy(),)
    if key == "3_6":
        return (csv("figure_3_6_heritability_24_traits.csv").to_dict("records"),)
    if key == "3_7":
        frame = csv("figure_3_7_trajectory_correlation.csv", index_col=0)
        correlation = frame.loc[module.TRAITS, module.TRAITS].to_numpy()
        links = csv("figure_3_7_average_linkage.csv").to_numpy()
        groups = csv("figure_3_7_cluster_assignments.csv").set_index("trait")
        order = [module.TRAITS.index(trait) for trait in groups.sort_values("display_order").index]
        return correlation, links, groups.loc[module.TRAITS, "cluster_k8"].to_numpy(), order
    if key == "3_8":
        selection = csv("figure_3_8_representative_selection.csv").to_dict("records")
        arrays, means = module.load_trajectories(selection)
        old = csv("figure_3_8_representative_mean_trajectories.csv")
        now = pd.DataFrame(means)
        pd.testing.assert_frame_equal(old[["trait", "time_coordinate"]], now[["trait", "time_coordinate"]])
        np.testing.assert_allclose(old["mean"], now["mean"], rtol=0, atol=1e-12)
        return selection, arrays
    if key == "3_9":
        return (csv("figure_3_9_simulated_phenotype_components.csv").to_dict("records"),)
    if key == "3_10":
        return ()
    if key == "3_11":
        frame = csv("figure_3_11_individual_split.csv")
        counts = frame.subset.value_counts().to_dict()
        assignment = np.array([module.SPLITS.index(value) for value in frame.subset])
        return assignment, counts
    if key == "5_1":
        return (csv(name + ".csv"),)
    if key in {"5_2", "5_3", "5_6", "5_8", "5_10"}:
        return (csv(name + "_summary.csv"),)
    if key == "5_4":
        return (csv(name + "_runs.csv"),)
    if key == "5_5":
        return csv(name + "_runs.csv"), csv(name + "_summary.csv")
    if key == "5_7":
        return (csv(name + "_ranks.csv"),)
    if key == "5_9":
        return (csv(name + ".csv"),)
    if key == "5_11":
        return csv(name + "_summary.csv"), protocol()["config"]["phenotype_names"]
    if key == "5_12":
        return csv(name + "_summary.csv"), protocol()
    if key == "5_13":
        return (csv(name + "_samples.csv", dtype={"sample_id": "string"}),
                csv(name + "_trajectories.csv", dtype={"sample_id": "string"}), protocol())
    if key == "5_14":
        return csv(name + "_pairs.csv"), csv(name + "_summary.csv"), protocol()["config"]["phenotype_names"]
    raise ValueError(f"No saved-data adapter for {key}")


def numeric_signature(fig: Figure) -> str:
    digest = hashlib.sha256()

    def add(value):
        arr = np.asanyarray(value)
        digest.update(str(arr.shape).encode())
        if np.ma.isMaskedArray(arr):
            digest.update(np.ma.getmaskarray(arr).tobytes())
            arr = arr.filled(np.nan)
        if arr.dtype.kind in "biufc":
            digest.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        else:
            digest.update(repr(arr.tolist()).encode())

    for ax in fig.axes:
        add(ax.get_xlim()); add(ax.get_ylim())
        for line in ax.lines:
            add(line.get_xdata()); add(line.get_ydata())
        for image in ax.images:
            add(image.get_array())
        for collection in ax.collections:
            add(collection.get_offsets())
            if collection.get_array() is not None:
                add(collection.get_array())
            if hasattr(collection, "get_segments"):
                for segment in collection.get_segments():
                    add(segment)
            else:
                for path in collection.get_paths():
                    add(path.vertices)
    return digest.hexdigest()


def axis_text_items(fig):
    values = [getattr(fig, name, None) for name in ["_supxlabel", "_supylabel"]]
    for ax in fig.axes:
        values.extend([ax.xaxis.label, ax.yaxis.label])
    values.extend(item for item in fig.texts if item.get_text() == AXIS_LABELS["time"])
    return [item for item in values if item is not None and item.get_visible() and item.get_text()]


def check_label_bounds(fig, *, allow_wrap: bool):
    fig.canvas.draw()
    wraps = []
    for item in axis_text_items(fig):
        box = item.get_window_extent(fig.canvas.get_renderer())
        outside = lambda b: b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.x1 + 1 or b.y1 > fig.bbox.y1 + 1
        if outside(box) and allow_wrap:
            words = re.findall(r"\$[^$]*\$|\S+", item.get_text())
            if len(words) > 1:
                original = item.get_text()
                split = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
                item.set_text(" ".join(words[:split]) + "\n" + " ".join(words[split:]))
                item.set_multialignment("center")
                fig.canvas.draw()
                box = item.get_window_extent(fig.canvas.get_renderer())
                wraps.append({"before": original, "after": item.get_text()})
        if outside(box):
            raise ValueError(f"Axis label extends outside the canvas: {item.get_text()}")
    return wraps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Refresh only figures listed in the saved edit plan")
    parser.add_argument("--baseline-commit", default="6fd6ef2")
    options = parser.parse_args()
    baseline = json.loads((REPORT / "修改前状态.json").read_text(encoding="utf-8"))
    edited = set(baseline["figure_keys"])
    saved_savefig = Figure.savefig
    results = []
    current = {}

    def capture(fig, target, *args, **kwargs):
        path = Path(target).resolve()
        current["signature"] = numeric_signature(fig)
        current["labels"] = figure_axis_labels(fig)
        if current["phase"] == "after":
            validate_figure_axis_labels(fig)
            current["wraps"] = check_label_bounds(fig, allow_wrap=True)
            current["labels"] = figure_axis_labels(fig)
            if options.refresh and current["key"] in edited:
                saved_savefig(fig, target, *args, **kwargs)
        current["figure"] = path.relative_to(ROOT).as_posix()

    Figure.savefig = capture
    try:
        paths = sorted((PROJECT / "scripts").glob("figure_*.py"),
                       key=lambda p: tuple(map(int, re.match(r"figure_(\d+)_(\d+)_", p.name).groups())))
        for path in paths:
            key = "_".join(re.match(r"figure_(\d+)_(\d+)_", path.name).groups())
            module = importlib.import_module(path.stem)
            arguments = inputs(key, module)
            before = None
            if key in edited:
                previous = subprocess.check_output(["git", "show", f"{options.baseline_commit}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT).decode("utf-8-sig")
                tree = ast.parse(previous)
                function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "draw")
                namespace = module.__dict__.copy()
                exec(compile(ast.Module(body=[copy.deepcopy(function)], type_ignores=[]), str(path), "exec"), namespace)
                current = {"key": key, "phase": "before"}
                namespace["draw"](*arguments)
                before = current.copy()
            current = {"key": key, "phase": "after"}
            module.draw(*arguments)
            if before and before["signature"] != current["signature"]:
                raise ValueError(f"Numerical artists or limits changed in figure {key}")
            record = current.copy()
            record["before_labels"] = before["labels"] if before else current["labels"]
            record["numerical_signature_unchanged"] = bool(before) if before else None
            record["refreshed"] = options.refresh and key in edited
            results.append(record)
            print(f"Checked {key}: {len(record['labels'])} axis labels; refreshed={record['refreshed']}", flush=True)
            metadata_path = DATA / (path.stem + ".json")
            if options.refresh and key in edited and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                metadata["axis_labels"] = record["labels"]
                if key == "5_2":
                    metadata["x_axis"] = AXIS_LABELS["training_sample_size"] + ", logarithmic scale base 2."
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        Figure.savefig = saved_savefig

    unchanged_csv = all((ROOT / name).exists() and sha(ROOT / name) == value
                        for name, value in baseline["csv_before_sha256"].items())
    if not unchanged_csv:
        raise ValueError("An existing plot-data CSV was modified")
    report = {"backup_commit": options.baseline_commit, "canonical_labels": AXIS_LABELS,
              "n_figures_checked": len(results), "n_figures_refreshed": sum(row["refreshed"] for row in results),
              "existing_csv_count": len(baseline["csv_before_sha256"]), "all_existing_csv_unchanged": unchanged_csv,
              "axis_labels_within_canvas": True, "results": results}
    (REPORT / "核查结果.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ["n_figures_checked", "n_figures_refreshed", "all_existing_csv_unchanged"]}), flush=True)


if __name__ == "__main__":
    main()
