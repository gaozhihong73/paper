"""Diagnose latent curve concordance and maximum-growth timing in simulation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

from dynags_figures.style import thesis_style

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]
BASE = ROOT / "05_实验结果" / "5.2"
DATA = PROJECT / "data"
STEM = "figure_5_5_latent_curve_diagnostics"
OUTPUT = ROOT / "07_论文图表" / "5-5_潜在轨迹一致性与主要增长时间恢复误差.png"
SIZES = [5000, 10000]
TRAIN = [100, 200, 400, 800, 1600]
H = [.1, .4, .8]
SETTINGS = {"Oracle": "experiment1_oracle", "Screened": "experiment2_screened"}
COLORS = {.1: "#0072B2", .4: "#D55E00", .8: "#009E73"}
METRICS = ["mean_ccc", "growth_time_mae"]
REFERENCE = "https://github.com/gaozhihong73/Code/blob/main/experiments/analysis/build_supplementary_tables.py"


def ccc(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Lin's CCC across the 20 observed times, per individual (population moments)."""
    pmean = prediction.mean(axis=1, keepdims=True)
    tmean = truth.mean(axis=1, keepdims=True)
    covariance = np.mean((prediction-pmean)*(truth-tmean), axis=1)
    denominator = prediction.var(axis=1) + truth.var(axis=1) + (pmean-tmean).ravel()**2
    result = np.full(len(truth), np.nan)
    np.divide(2*covariance, denominator, out=result, where=denominator > 1e-12)
    return result


def growth_time(values: np.ndarray, times: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of repository _inflection, including invalid flags."""
    dense = np.linspace(float(times[0]), float(times[-1]), 1000)
    margin = .05 * float(times[-1]-times[0])
    internal = np.flatnonzero((dense >= times[0]+margin) & (dense <= times[-1]-margin))
    assert len(internal) >= 3
    derivative = PchipInterpolator(times, values, axis=1, extrapolate=False).derivative()(dense)
    location = np.argmax(derivative[:, internal], axis=1)
    index = internal[location]
    rate = derivative[np.arange(len(values)), index]
    status = np.full(len(values), "valid", dtype=object)
    status[~np.isfinite(rate) | (rate <= 0)] = "nonpositive_maximum_derivative"
    status[(location == 0) | (location == len(internal)-1)] = "maximum_derivative_on_internal_boundary"
    status[np.ptp(values, axis=1) < .01*scale] = "amplitude_below_one_percent_train_scale"
    peak = dense[index].copy()
    peak[status != "valid"] = np.nan
    return peak, status


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    individual_frames = []
    run_rows = []
    for size in SIZES:
        for n in TRAIN:
            for h in H:
                condition = f"P{size}_NTR{n}_H{h}"
                for repeat in range(1, 11):
                    relative = Path(condition) / f"rep_{repeat:02d}"
                    source = BASE / "experiments" / relative
                    observed = pd.read_csv(source / "sim_pheno.csv", dtype={"SampleID": str})
                    latent = pd.read_csv(source / "truth/true_curve.csv", dtype={"SampleID": str})
                    config = json.loads((source / "simulation_config.json").read_text(encoding="utf-8"))
                    columns = config["time_columns"]
                    times = np.array(config["time_coordinates"], dtype=float)
                    assert observed.SampleID.tolist() == latent.SampleID.tolist()
                    with np.load(source / "splits/split_seed73.npz") as split:
                        train = split["train_idx"]
                        test = split["test_idx"]
                    assert len(train) == n
                    y = observed[columns].to_numpy(float)
                    truth = latent.iloc[test][columns].to_numpy(float)
                    scale = float(np.percentile(y[train], 95) - np.percentile(y[train], 5))
                    assert np.isfinite(scale) and scale > 0
                    true_time, true_status = growth_time(truth, times, scale)
                    for setting, folder in SETTINGS.items():
                        prediction_file = BASE / folder / relative / "test_predictions.npz"
                        with np.load(prediction_file) as archive:
                            assert np.array_equal(archive["sample_idx"], test)
                            assert np.array_equal(archive["time_coordinates"], times)
                            assert np.allclose(archive["targets"][:, 0, :], y[test], rtol=0, atol=5e-6)
                            prediction = archive["predictions"][:, 0, :].astype(float)
                        assert prediction.shape == truth.shape and np.isfinite(prediction).all()
                        agreement = ccc(prediction, truth)
                        peak, status = growth_time(prediction, times, scale)
                        valid = np.isfinite(peak) & np.isfinite(true_time)
                        error = peak - true_time
                        fields = {"condition": condition, "snps": size, "n_train": n,
                                  "heritability": h, "repeat": repeat, "setting": setting}
                        individual_frames.append(pd.DataFrame(fields | {
                            "sample_index": test, "sample_id": observed.iloc[test].SampleID.to_numpy(),
                            "ccc": agreement, "true_growth_time": true_time,
                            "predicted_growth_time": peak, "true_time_status": true_status,
                            "predicted_time_status": status, "valid_time_pair": valid,
                            "growth_time_error": error, "growth_time_abs_error": np.abs(error)}))
                        run_rows.append(fields | {"n_test": len(test),
                            "ccc_valid_count": int(np.isfinite(agreement).sum()),
                            "mean_ccc": float(np.nanmean(agreement)),
                            "growth_time_valid_count": int(valid.sum()),
                            "growth_time_valid_fraction": float(valid.mean()),
                            "growth_time_mae": float(np.nanmean(np.abs(error))) if valid.any() else np.nan,
                            "growth_time_mean_bias": float(np.nanmean(error)) if valid.any() else np.nan,
                            "train_p95_p5": scale,
                            "prediction_file": str(prediction_file.relative_to(ROOT))})
        print(f"Finished SNP size {size:,}", flush=True)
    individuals = pd.concat(individual_frames, ignore_index=True)
    runs = pd.DataFrame(run_rows)
    assert len(runs) == 600 and not runs.duplicated(["condition", "repeat", "setting"]).any()
    assert np.isfinite(runs[METRICS].to_numpy()).all()
    return individuals, runs


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    # Identical resample rows across settings preserve the matched-repeat structure.
    resamples = np.random.default_rng(73).integers(0, 10, size=(10000, 10))
    rows = []
    keys = ["snps", "n_train", "heritability", "setting"]
    for values, group in runs.groupby(keys):
        group = group.sort_values("repeat")
        assert len(group) == 10 and group.repeat.tolist() == list(range(1, 11))
        for metric in METRICS:
            x = group[metric].to_numpy()
            low, high = np.quantile(x[resamples].mean(axis=1), [.025, .975])
            rows.append(dict(zip(keys, values)) | {"metric": metric, "n_repeats": len(x),
                         "mean": x.mean(), "sd": x.std(ddof=1),
                         "ci95_lower": low, "ci95_upper": high})
    return pd.DataFrame(rows)


def draw(runs: pd.DataFrame, summary: pd.DataFrame) -> dict:
    styles = {"Oracle": {"linestyle": (0, (4, 2)), "marker": "o"},
              "Screened": {"linestyle": "-", "marker": "s"}}
    limits = {}
    with thesis_style(base_font_size=8):
        plt.rcParams["axes.unicode_minus"] = True
        fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.9), sharex=True, sharey="row")
        for i, metric in enumerate(METRICS):
            # Include all repeat-level observations and CI endpoints in each row's limits.
            candidates = np.r_[runs[metric], summary[summary.metric == metric].ci95_lower,
                               summary[summary.metric == metric].ci95_upper]
            lower, upper = float(candidates.min()), float(candidates.max())
            margin = max((upper-lower)*.1, .0001)
            ylim = (lower-margin, min(1.001, upper+margin)) if metric == "mean_ccc" else (0., upper+margin)
            limits[metric] = list(ylim)
            for j, size in enumerate(SIZES):
                ax = axes[i, j]
                for h in H:
                    for setting in SETTINGS:
                        part = summary[(summary.snps == size) & (summary.metric == metric) &
                                       (summary.heritability == h) & (summary.setting == setting)]
                        part = part.sort_values("n_train")
                        ax.fill_between(part.n_train, part.ci95_lower, part.ci95_upper,
                                        color=COLORS[h], alpha=.07, linewidth=0, zorder=1)
                        raw = runs[(runs.snps == size) & (runs.heritability == h) & (runs.setting == setting)]
                        ax.scatter(raw.n_train, raw[metric], s=3, color=COLORS[h],
                                   alpha=.16, linewidths=0, zorder=2)
                        ax.plot(part.n_train, part["mean"], color=COLORS[h], lw=1.1,
                                markerfacecolor="white" if setting == "Oracle" else COLORS[h],
                                markeredgewidth=.7, markersize=3, **styles[setting], zorder=4)
                ax.set_xscale("log", base=2)
                ax.set_xlim(85, 1880)
                ax.set_xticks(TRAIN, labels=[str(n) for n in TRAIN])
                ax.minorticks_off()
                ax.set_ylim(*ylim)
                ax.set_title(f"{'ABCD'[i*2+j]}  {size:,} SNPs", loc="left", fontsize=8,
                             color="#193C51", pad=6)
                ax.tick_params(length=2.5, pad=2, labelsize=7)
                ax.grid(axis="y", color="#DFE7EB", lw=.5)
                ax.spines[["left", "bottom"]].set_color("#A7B8C0")
                if j == 0:
                    ax.set_ylabel("Curve CCC" if i == 0 else "Growth-time MAE (time units)",
                                  fontsize=8, labelpad=6)
        fig.supxlabel("Training individuals", y=.165, fontsize=8)
        fig.legend(handles=[Line2D([], [], color=COLORS[h], lw=1.6, label=rf"$h^2={h}$") for h in H],
                   loc="lower center", bbox_to_anchor=(.5, .087), ncol=3, fontsize=7.5,
                   columnspacing=2.8, handlelength=2.1)
        fig.legend(handles=[Line2D([], [], color="#455560", lw=1.1, markersize=3,
                                   markerfacecolor="white" if s == "Oracle" else "#455560",
                                   **styles[s], label=s+" SNPs") for s in SETTINGS],
                   loc="lower center", bbox_to_anchor=(.5, .033), ncol=2, fontsize=7.5,
                   columnspacing=3.4, handlelength=2.8)
        fig.subplots_adjust(left=.115, right=.98, top=.94, bottom=.25, wspace=.17, hspace=.34)
        fig.savefig(OUTPUT, dpi=600, facecolor="white")
        plt.close(fig)
    return limits


if __name__ == "__main__":
    individuals, runs = collect()
    summary = summarize(runs)
    DATA.mkdir(parents=True, exist_ok=True)
    for name, frame in [("individuals", individuals), ("runs", runs), ("summary", summary)]:
        frame.to_csv(DATA / f"{STEM}_{name}.csv", index=False, encoding="utf-8-sig")
    limits = draw(runs, summary)
    metadata = {"reference_code": REFERENCE,
                "reference_blob_sha": "10a6660e2d37412d584629434846dc63f1855cb3",
                "ccc": "Per-individual Lin CCC using population moments across the 20 saved "
                       "observation times; reference is the latent noise-free curve.",
                "growth_time": "PCHIP interpolation to 1000 equally spaced time coordinates. "
                               "Maximum first derivative within the interior 90% of the full time range. "
                               "Invalid if amplitude < 1% of training-observation P95-P5, maximum lies "
                               "on the interior boundary, or maximum derivative is nonpositive.",
                "aggregation": "First mean individual CCC and mean absolute valid growth-time error "
                               "within each repeat; then equal-weight means over 10 independent repeats. "
                               "Timing MAE includes only valid predicted/latent time pairs.",
                "uncertainty": "Percentile 95% confidence interval from 10000 repeat-level bootstrap "
                               "resamples; seed 73. Pale dots are repeat-level summaries.",
                "n_runs": len(runs), "n_individual_curves": len(individuals),
                "invalid_time_pairs": int((~individuals.valid_time_pair).sum()),
                "axis_limits": limits, "figure": str(OUTPUT.relative_to(ROOT))}
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(runs.groupby("setting")[METRICS].agg(["min", "max"]).to_string())
    print("Invalid time pairs:", metadata["invalid_time_pairs"], "of", len(individuals))
    print(OUTPUT)
