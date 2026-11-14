"""Plot the components of one saved longitudinal phenotype simulation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dynags_figures.style import thesis_style
from dynags_figures.labels import AXIS_LABELS
from dynags_figures.layout import place_shared_ylabel

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "04_数据" / "3-4_P10000_NTR1600_H0.8_rep_08"
DATA = PROJECT / "data" / "figure_3_9_simulated_phenotype_components.csv"
FIGURE = ROOT / "07_论文图表" / "3-9_模拟纵向表型的组成与生成示例.png"


def load_components() -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    config = json.loads((SOURCE / "simulation_config.json").read_text(encoding="utf-8"))
    times = np.asarray(config["time_coordinates"], dtype=float)
    expected_columns = ["SampleID", *config["time_columns"]]
    ids = pd.read_csv(SOURCE / "sample_ids.csv", dtype={"SampleID": str})["SampleID"].tolist()
    if len(ids) != config["n_samples"] or len(set(ids)) != len(ids):
        raise ValueError("Simulation sample IDs are incomplete or duplicated")

    arrays = {}
    for name, relative in (
        ("observed", "sim_pheno.csv"),
        ("noiseless", "truth/true_curve.csv"),
        ("genetic", "truth/true_gv.csv"),
    ):
        frame = pd.read_csv(SOURCE / relative, dtype={"SampleID": str})
        if list(frame.columns) != expected_columns or frame["SampleID"].tolist() != ids:
            raise ValueError(f"Sample IDs or time columns differ in {relative}")
        values = frame.iloc[:, 1:].to_numpy(dtype=float)
        if values.shape != (len(ids), len(times)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid component matrix: {relative}")
        arrays[name] = values

    baseline_matrix = arrays["noiseless"] - arrays["genetic"]
    baseline = baseline_matrix.mean(axis=0)
    if not np.allclose(baseline_matrix, baseline[None, :], atol=1e-6, rtol=0):
        raise ValueError("Noiseless phenotype does not decompose into a common baseline and GV")
    arrays["baseline"] = baseline
    arrays["residual"] = arrays["observed"] - arrays["noiseless"]
    return times, arrays, len(ids)


def summarize(times: np.ndarray, arrays: dict[str, np.ndarray], n: int) -> list[dict]:
    summary = {"baseline": arrays["baseline"]}
    for name in ("genetic", "residual", "observed"):
        values = arrays[name]
        summary[f"{name}_mean"] = values.mean(axis=0)
        summary[f"{name}_q25"] = np.quantile(values, 0.25, axis=0)
        summary[f"{name}_q75"] = np.quantile(values, 0.75, axis=0)
    return [
        {
            "scenario": "P10000_NTR1600_H0.8",
            "repetition": 8,
            "n_individuals": n,
            "time_coordinate": int(time),
            **{name: float(values[index]) for name, values in summary.items()},
        }
        for index, time in enumerate(times)
    ]


def save_data(rows: list[dict]) -> None:
    DATA.parent.mkdir(parents=True, exist_ok=True)
    with DATA.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(rows: list[dict]) -> None:
    times = np.array([row["time_coordinate"] for row in rows])
    values = {key: np.array([row[key] for row in rows]) for key in rows[0] if key not in
              {"scenario", "repetition", "n_individuals", "time_coordinate"}}
    with thesis_style(base_font_size=8.0):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.8), sharex=True, dpi=300)
        colors = {"baseline": "#176B91", "genetic": "#0072B2",
                  "residual": "#D55E00", "observed": "#009E73"}
        panels = [
            ("A  Logistic baseline", "baseline", axes[0, 0]),
            ("B  Genetic contribution", "genetic", axes[0, 1]),
            ("C  Random residual", "residual", axes[1, 0]),
            ("D  Observed phenotype", "observed", axes[1, 1]),
        ]
        for title, name, ax in panels:
            color = colors[name]
            if name == "baseline":
                ax.plot(times, values["baseline"], color=color, linewidth=1.75, zorder=3)
                ax.set_ylim(0, 34)
            else:
                ax.fill_between(times, values[f"{name}_q25"], values[f"{name}_q75"],
                                color=color, alpha=0.20, linewidth=0, zorder=1)
                ax.plot(times, values[f"{name}_mean"], color=color, linewidth=1.75,
                        zorder=3)
                if name in {"genetic", "residual"}:
                    ax.axhline(0, color="#8CA4AF", linewidth=0.7, linestyle="--", zorder=0)
                if name == "observed":
                    ax.plot(times, values["baseline"], color="#647986", linewidth=1.1,
                            linestyle=(0, (4, 2)), zorder=2)
                    ax.set_ylim(0, 36)
            ax.set_title(title, loc="left", color="#193C51", fontsize=8.2, pad=5)
            ax.set_xlim(float(times.min()), float(times.max()))
            ax.set_xticks([1, 5, 10, 15, 20])
            ax.tick_params(axis="both", labelsize=7.2, length=2.5, pad=2)
            ax.grid(axis="y", color="#DFE7EB", linewidth=0.5)
            ax.spines["left"].set_color("#A7B8C0")
            ax.spines["bottom"].set_color("#A7B8C0")

        fig.supxlabel(AXIS_LABELS["time"], y=0.082, fontsize=8)
        fig.legend(
            handles=[
                Line2D([], [], color="#3D5563", lw=1.7,
                       label="Baseline / component mean"),
                Patch(facecolor="#A5C1CF", edgecolor="none",
                      label="25th–75th percentile"),
                Line2D([], [], color="#647986", lw=1.1, ls="--",
                       label="Baseline in panel D"),
            ],
            loc="lower center", bbox_to_anchor=(0.5, 0.014), ncol=3,
            fontsize=7.0, frameon=False,
        )
        fig.subplots_adjust(left=0.12, right=0.98, top=0.945, bottom=0.16,
                            wspace=0.30, hspace=0.42)
        place_shared_ylabel(fig, axes, AXIS_LABELS["phenotypic_value"])
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE, dpi=300, facecolor="white", bbox_inches="tight",
                    pad_inches=0.13)
        plt.close(fig)


if __name__ == "__main__":
    time, components, count = load_components()
    data = summarize(time, components, count)
    save_data(data)
    draw(data)
    print(f"Scenario: P10000_NTR1600_H0.8; repetition: 8; individuals: {count}")
    print(f"Data: {DATA}")
    print(f"Figure: {FIGURE}")
