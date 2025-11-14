"""
Classification metrics computation and aggregation.

This module provides functions to compute comprehensive classification metrics
including accuracy, precision, recall, F1, AUC-ROC, and confusion matrices.
"""

import logging
from typing import Dict, List, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1)
        y_proba: Predicted probabilities for positive class (optional, for AUC)
    
    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Positive predictive value
            - recall: Sensitivity/True positive rate
            - specificity: True negative rate
            - f1: F1 score
            - auc_roc: Area under ROC curve (if y_proba provided)
            - confusion_matrix: 2x2 confusion matrix as list
            - true_positives, false_positives, true_negatives, false_negatives
            - support: Number of samples per class
    
    Example:
        >>> metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        >>> print(f"F1 Score: {metrics['f1']:.3f}")
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "specificity": None,  # Computed from confusion matrix below
    }
    
    # Compute AUC if probabilities available
    if y_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            # Handle case where only one class present in y_true
            metrics["auc_roc"] = None
            logger.warning("AUC-ROC could not be computed (only one class in y_true)")
    else:
        metrics["auc_roc"] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Compute specificity and breakdown from confusion matrix
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)
    
    # Support (number of samples per class)
    unique, counts = np.unique(y_true, return_counts=True)
    metrics["support"] = dict(zip(unique.tolist(), counts.tolist()))
    
    return metrics


def aggregate_fold_metrics(fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute mean and standard deviation of metrics across folds.
    
    Args:
        fold_results: List of dictionaries containing metrics for each fold
    
    Returns:
        Dictionary with mean and std for each metric, plus total confusion matrix
    
    Example:
        >>> aggregated = aggregate_fold_metrics(fold_results)
        >>> print(f"Accuracy: {aggregated['accuracy_mean']:.3f} ± {aggregated['accuracy_std']:.3f}")
    """
    aggregated = {}
    
    # Metrics to aggregate
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc"]
    
    for metric in metric_names:
        values = [
            fold[metric] 
            for fold in fold_results 
            if fold.get(metric) is not None
        ]
        
        if values:
            aggregated[f"{metric}_mean"] = np.mean(values)
            aggregated[f"{metric}_std"] = np.std(values)
        else:
            aggregated[f"{metric}_mean"] = None
            aggregated[f"{metric}_std"] = None
    
    # Overall confusion matrix (sum across folds)
    total_cm = np.zeros((2, 2), dtype=int)
    for fold in fold_results:
        cm = np.array(fold["confusion_matrix"])
        if cm.shape == (2, 2):
            total_cm += cm
    
    aggregated["confusion_matrix_total"] = total_cm.tolist()
    
    return aggregated


def log_metrics(metrics: Dict[str, Any], prefix: str = "") -> None:
    """
    Log metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for log messages (e.g., "Fold 1: ")
    
    Example:
        >>> log_metrics(metrics, prefix="Fold 1: ")
    """
    if prefix:
        logger.info(f"{prefix}")
    
    # Log main metrics
    for metric_name in ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc"]:
        if metric_name in metrics and metrics[metric_name] is not None:
            logger.info(f"  {metric_name.capitalize():12s}: {metrics[metric_name]:.4f}")


def log_aggregated_metrics(aggregated: Dict[str, Any]) -> None:
    """
    Log aggregated metrics with mean ± std format.
    
    Args:
        aggregated: Dictionary with aggregated metrics (from aggregate_fold_metrics)
    
    Example:
        >>> log_aggregated_metrics(aggregated_metrics)
    """
    logger.info("\nAggregated Results (Mean ± Std):")
    
    metric_names = ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc"]
    
    for metric in metric_names:
        mean = aggregated.get(f"{metric}_mean")
        std = aggregated.get(f"{metric}_std")
        
        if mean is not None:
            logger.info(f"  {metric.capitalize():12s}: {mean:.4f} ± {std:.4f}")
