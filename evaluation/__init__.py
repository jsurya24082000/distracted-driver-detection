"""Evaluation module for metrics, efficiency analysis, domain generalization, and CAM quality."""

from .metrics import compute_metrics, plot_confusion_matrix, plot_training_curves
from .efficiency import count_flops, count_params, measure_latency, efficiency_report
from .domain_generalization import DomainGeneralizationEvaluator, compare_models_domain_gap
from .cam_quality import CAMQualityEvaluator, evaluate_all_models_cam_quality

__all__ = [
    "compute_metrics",
    "plot_confusion_matrix",
    "plot_training_curves",
    "count_flops",
    "count_params",
    "measure_latency",
    "efficiency_report",
    "DomainGeneralizationEvaluator",
    "compare_models_domain_gap",
    "CAMQualityEvaluator",
    "evaluate_all_models_cam_quality",
]
