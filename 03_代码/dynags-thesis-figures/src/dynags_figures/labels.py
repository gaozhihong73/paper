"""Project-wide axis terminology, separate from experiment data and formulas."""

from __future__ import annotations

AXIS_LABELS = {
    "time": "Time",
    "phenotypic_value": "Phenotypic value",
    "mean_phenotypic_value": "Mean phenotypic value",
    "training_sample_size": "Training sample size",
    "pcc": "PCC",
    "r2_time": r"$R^2_{\mathrm{time}}$",
    "rmse": "RMSE",
    "delta_pcc": r"$\Delta\mathrm{PCC}$",
    "delta_r2_time": r"$\Delta R^2_{\mathrm{time}}$",
    "delta_rmse": r"$\Delta\mathrm{RMSE}$",
    "prediction_error": "Prediction error",
    "repeat": "Repeat",
    "retained_variants": "Number of retained variants (log scale)",
    "estimated_heritability": r"Heritability ($H^2$)",
    "simulation_heritability": r"Heritability ($h^2$)",
}

_ALIASES = {
    "Time coordinate": "time", "Time coordinates": "time",
    "Time point": "time", "Time points": "time", "Timepoint": "time",
    "Trait value (input scale)": "phenotypic_value",
    "Phenotype value": "phenotypic_value",
    "Phenotype value (simulation scale)": "phenotypic_value",
    "Value (simulation scale)": "phenotypic_value",
    "Mean phenotype value": "mean_phenotypic_value",
    "Training individuals": "training_sample_size",
    "RMSE (simulation scale)": "rmse",
    "Prediction − observed": "prediction_error",
    "Independent repeat": "repeat",
    "Retained variants (log scale)": "retained_variants",
    "Heritability estimate (H2)": "estimated_heritability",
}
_NORMALIZED_ALIASES = {" ".join(text.split()).casefold(): key for text, key in _ALIASES.items()}


def suggested_axis_label(text: str) -> str | None:
    """Return a replacement only for a known equivalent axis label."""
    key = _NORMALIZED_ALIASES.get(" ".join(text.split()).casefold())
    return AXIS_LABELS[key] if key else None


def figure_axis_labels(fig) -> list[dict[str, str]]:
    """Read rendered axis/colorbar titles and shared figure-wide axis labels."""
    records = []
    for index, ax in enumerate(fig.axes):
        for role, text in [("x", ax.get_xlabel()), ("y", ax.get_ylabel())]:
            if text:
                records.append({"artist": f"axes[{index}].{role}", "text": text})
    for role in ["_supxlabel", "_supylabel"]:
        item = getattr(fig, role, None)
        if item is not None and item.get_text():
            records.append({"artist": role, "text": item.get_text()})
    for index, item in enumerate(fig.texts):
        if item is getattr(fig, "_supxlabel", None) or item is getattr(fig, "_supylabel", None):
            continue
        if item.get_text() == AXIS_LABELS["time"] or suggested_axis_label(item.get_text()):
            records.append({"artist": f"figure.texts[{index}]", "text": item.get_text()})
    return records


def validate_figure_axis_labels(fig) -> list[dict[str, str]]:
    records = figure_axis_labels(fig)
    errors = [(row["text"], suggested_axis_label(row["text"])) for row in records
              if suggested_axis_label(row["text"])]
    if errors:
        raise ValueError(f"Noncanonical axis labels: {errors}")
    return records
