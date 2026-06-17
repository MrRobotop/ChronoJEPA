"""Linear and kNN probes, forecasting, Mahalanobis anomaly scoring, label-free selection."""

from .anomaly import MahalanobisScorer, inject_anomaly
from .collapse import across_time_variance, collapse_report, effective_rank
from .comparison import (
    format_anomaly_table,
    format_classification_table,
    format_comparison_table,
    format_horizon_table,
    format_lambda_table,
    format_multiseed_table,
    run_anomaly_comparison,
    run_classification_comparison,
    run_horizon_sweep,
    run_lambda_sweep,
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
    "format_anomaly_table",
    "format_classification_table",
    "format_comparison_table",
    "format_horizon_table",
    "format_lambda_table",
    "format_multiseed_table",
    "inject_anomaly",
    "knn_probe",
    "label_free_model_selection",
    "linear_probe",
    "run_anomaly_comparison",
    "run_classification_comparison",
    "run_horizon_sweep",
    "run_lambda_sweep",
    "run_multiseed_comparison",
    "run_placement_comparison",
    "selection_report",
]
