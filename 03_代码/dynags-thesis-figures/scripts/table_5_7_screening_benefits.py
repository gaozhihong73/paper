"""Export the supplied screening benefits as editable thesis Table 5-7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from figure_5_10_screening_benefits import MODELS, SOURCE, collect

PROJECT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT / "data"
STEM = "table_5_7_screening_benefits"
FIGURE_DATA = DATA / "figure_5_10_screening_benefits_summary.csv"
OUTPUT = ROOT / "07_论文图表/表5-7_同一模型筛选前后的配对预测收益.md"
TITLE = "表 5-7 同一模型筛选前后的配对预测收益"
NOTE = (
    "注：数值为 10 次独立重复中同一模型配对差值的均值，方括号内为 95% 置信区间。"
    "ΔPCC 和 $\\Delta R^2_{\\mathrm{time}}$ 为筛选后减筛选前，ΔRMSE 为筛选前减筛选后，"
    "正值均表示筛选带来预测收益。筛选前输入完整的 10,000 个 SNP。"
    "Ridge 表示仅采用加性剂量编码的岭回归，该行比较完整 SNP 与筛选 SNP 两种输入。"
)


def cell(row: pd.Series) -> str:
    return (f"{row['mean_difference']:.3f} "
            f"[{row['ci95_lower']:.3f}, {row['ci95_upper']:.3f}]")


def main() -> None:
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    data = collect()
    figure_data = pd.read_csv(FIGURE_DATA)
    pd.testing.assert_frame_equal(data, figure_data, check_dtype=False, check_exact=True)
    if data["n_repeats"].tolist() != [10] * 21:
        raise ValueError("Expected the supplied 10-repeat summaries for all 21 entries")

    indexed = data.set_index(["model", "metric"])
    lines = [TITLE, "",
             "| 模型 | ΔPCC（95% CI） | $\\Delta R^2_{\\mathrm{time}}$（95% CI） | ΔRMSE（95% CI） |",
             "| --- | ---: | ---: | ---: |"]
    for model in MODELS:
        values = [cell(indexed.loc[(model, metric)]) for metric in ["pcc", "r2", "rmse"]]
        lines.append("| " + " | ".join([model, *values]) + " |")
    lines.extend(["", NOTE, ""])
    content = "\n".join(lines)
    if any(code in content for code in ["E1", "E2", "E3", "E4", "E5"]):
        raise ValueError("Internal experiment labels must not appear in the formal table")

    output_data = data.rename(columns={"metric": "source_metric"}).copy()
    output_data.insert(2, "metric", output_data.source_metric.replace({"r2": "r2_time"}))
    metadata = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_hash,
        "source_section": "Table E5a. Effect of SNP screening",
        "figure_data": str(FIGURE_DATA.relative_to(ROOT)),
        "figure_data_sha256": hashlib.sha256(FIGURE_DATA.read_bytes()).hexdigest(),
        "checks": "All 21 supplied means and intervals exactly match the data used in Figure 5-10.",
        "n_models": len(MODELS), "n_metrics": 3, "n_repeats": 10,
        "statistics": "Reported paired mean differences and 95% confidence intervals; copied without recalculation.",
        "display_precision": 3,
        "directions": {"pcc": "PCC_screened minus PCC_all",
                       "r2_time": "R2_time_screened minus R2_time_all",
                       "rmse": "RMSE_all minus RMSE_screened"},
        "model_labels": {"Ridge": "Ridge"},
        "caption": TITLE, "note": NOTE,
        "output": str(OUTPUT.relative_to(ROOT)),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(DATA / f"{STEM}_summary.csv", index=False, encoding="utf-8-sig")
    (DATA / f"{STEM}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT.write_text(content, encoding="utf-8")
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != source_hash:
        raise ValueError("The supplied experiment summary changed during table export")
    print(content)
    print(json.dumps({"models": len(MODELS), "numeric_entries": len(data),
                      "matches_figure_5_10": True, "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
