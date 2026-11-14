"""Summarize supplied overall metrics for thesis Table 5-6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from figure_5_6_paired_model_comparison import MODELS, collect

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
STEM = "table_5_6_overall_model_performance"


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        part = data[data.model == model].sort_values("repeat")
        assert part.repeat.tolist() == list(range(1, 11))
        row = {"model": model, "n_repeats": len(part)}
        for metric in ["pcc", "r2_time", "rmse"]:
            values = part[metric].dropna().to_numpy(float)
            assert len(values) == 10 or (model == "Mean" and metric == "pcc" and len(values) == 0)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) else np.nan
            row[f"{metric}_count"] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    runs = collect()
    summary = summarize(runs)
    DATA.mkdir(parents=True, exist_ok=True)
    runs.to_csv(DATA / f"{STEM}_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(DATA / f"{STEM}_summary.csv", index=False, encoding="utf-8-sig")
    values = summary.set_index("model")
    comparisons = {"baseline": "XGBoost",
        "pcc_difference": float(values.loc["DynaGS", "pcc_mean"]-values.loc["XGBoost", "pcc_mean"]),
        "r2_difference": float(values.loc["DynaGS", "r2_time_mean"]-values.loc["XGBoost", "r2_time_mean"]),
        "rmse_reduction": float(values.loc["XGBoost", "rmse_mean"]-values.loc["DynaGS", "rmse_mean"]),
        "relative_rmse_reduction_percent": float(100*(1-values.loc["DynaGS", "rmse_mean"]/
                                                      values.loc["XGBoost", "rmse_mean"]))}
    metadata = {"source": "Supplied 5.3 metrics.json overall fields, read directly via "
                          "figure_5_6_paired_model_comparison.collect.",
                "condition": {"snps": 10000, "n_train": 400, "heritability": .8},
                "aggregation": "Equal-weight mean and sample SD (ddof=1) over 10 independent repeats.",
                "missing": "Mean baseline PCC is undefined and displayed as N/A.",
                "display_precision": 3, "comparisons": comparisons}
    (DATA / f"{STEM}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.round(6).to_string(index=False))
    print(json.dumps(comparisons, indent=2))
