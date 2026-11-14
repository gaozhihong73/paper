"""Build reference-condition statistics for thesis Table 5-5."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
SOURCE = DATA / "figure_5_2_simulation_performance_runs.csv"
STEM = "table_5_5_paired_input_performance"


def build() -> pd.DataFrame:
    data = pd.read_csv(SOURCE)
    data = data[(data.snps == 10000) & (data.n_train == 400) & (data.heritability == .8)]
    assert len(data) == 20
    resamples = np.random.default_rng(73).integers(0, 10, size=(10000, 10))
    rows = []
    for metric in ["pcc", "r2_time", "rmse"]:
        pair = data.pivot(index="repeat", columns="setting", values=metric).sort_index()
        assert pair.index.tolist() == list(range(1, 11))
        assert np.isfinite(pair.to_numpy()).all()
        delta = (pair.Screened-pair.Oracle) * (-1 if metric == "rmse" else 1)
        low, high = np.quantile(delta.to_numpy()[resamples].mean(axis=1), [.025, .975])
        rows.append({"metric": metric, "n_pairs": len(pair),
                     "oracle_mean": pair.Oracle.mean(), "oracle_sd": pair.Oracle.std(ddof=1),
                     "screened_mean": pair.Screened.mean(), "screened_sd": pair.Screened.std(ddof=1),
                     "paired_difference": delta.mean(), "paired_sd": delta.std(ddof=1),
                     "ci95_lower": low, "ci95_upper": high})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = build()
    results.to_csv(DATA / f"{STEM}.csv", index=False, encoding="utf-8-sig")
    metadata = {"source": SOURCE.name,
                "condition": {"snps": 10000, "n_train": 400, "heritability": .8},
                "n_repeats": 10, "sd": "Sample SD across independent repeats (ddof=1).",
                "difference": {"pcc": "Screened minus Oracle", "r2_time": "Screened minus Oracle",
                               "rmse": "Oracle minus Screened"},
                "ci": "Percentile 95% confidence interval of mean paired differences; "
                      "10000 paired bootstrap resamples, random seed 73.",
                "rounding": "Compute with full precision and round only displayed values to 3 decimals."}
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(results.round(6).to_string(index=False))
