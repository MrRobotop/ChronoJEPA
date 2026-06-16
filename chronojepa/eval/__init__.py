"""Linear and kNN probes, forecasting, Mahalanobis anomaly scoring, label-free selection."""

from .anomaly import MahalanobisScorer
from .collapse import across_time_variance, collapse_report, effective_rank
from .comparison import (
    format_comparison_table,
    format_multiseed_table,
    run_multiseed_comparison,
    run_placement_comparison,
)
from .model_selection import label_free_model_selection, selection_report
from .probes import extract_features, forecast_linear_probe, knn_probe, linear_probe

__all__ = [
    "MahalanobisScorer",
    "across_time_variance",
    "collapse_report",
    "effective_rank",
    "extract_features",
    "forecast_linear_probe",
    "format_comparison_table",
    "format_multiseed_table",
    "knn_probe",
    "label_free_model_selection",
    "linear_probe",
    "run_multiseed_comparison",
    "run_placement_comparison",
    "selection_report",
]
