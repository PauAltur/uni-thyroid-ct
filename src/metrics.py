"""
Classification metrics computation utilities.

This module provides comprehensive metrics for binary and multiclass classification,
including accuracy, precision, recall, F1, specificity, AUC-ROC, and confusion matrix.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
)
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "binary",
    pos_label: int = 1,
    class_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, needed for AUC)
        average: Averaging strategy for multiclass ('binary', 'micro', 'macro', 'weighted')
        pos_label: Positive class label (for binary classification)
        class_labels: Array of class labels in the order they appear in y_prob columns
                      (typically from classifier.classes_). If None, assumes labels are
                      in sorted order [0, 1] for binary or sorted unique values for multiclass.
        
    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    
    # Precision, Recall, F1
    metrics["precision"] = float(precision_score(
        y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
    ))
    metrics["recall"] = float(recall_score(
        y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
    ))
    metrics["f1"] = float(f1_score(
        y_true, y_pred, average=average, pos_label=pos_label, zero_division=0
    ))
    
    # Sensitivity (same as recall)
    metrics["sensitivity"] = metrics["recall"]
    
    # Specificity (for binary classification)
    unique_labels = np.unique(y_true)
    if len(unique_labels) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=unique_labels).ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        metrics["true_positives"] = int(tp)
        
        # Positive and Negative Predictive Values
        metrics["ppv"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0  # Same as precision
        metrics["npv"] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    else:
        # Multiclass - don't store confusion matrix in metrics dict (causes DataFrame issues)
        # It can be computed separately if needed
        metrics["specificity"] = None
    
    # AUC-ROC (if probabilities provided)
    if y_prob is not None:
        try:
            if len(unique_labels) == 2:
                # Binary classification
                if y_prob.ndim == 2:
                    # Find the column index for the positive class
                    if class_labels is not None:
                        # Use provided class labels to find positive class column
                        pos_class_idx = np.where(class_labels == pos_label)[0]
                        if len(pos_class_idx) > 0:
                            y_prob_pos = y_prob[:, pos_class_idx[0]]
                        else:
                            logger.warning(f"Positive label {pos_label} not found in class_labels {class_labels}, using column 1")
                            y_prob_pos = y_prob[:, 1]
                    else:
                        # Assume sorted order: if pos_label is 1 and labels are [0,1], use column 1
                        # If labels are [1,0], this will be wrong - should provide class_labels
                        sorted_labels = np.sort(unique_labels)
                        pos_class_idx = np.where(sorted_labels == pos_label)[0]
                        if len(pos_class_idx) > 0:
                            y_prob_pos = y_prob[:, pos_class_idx[0]]
                        else:
                            logger.warning("Could not determine positive class column, using column 1")
                            y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob_pos))
            else:
                # Multiclass - use one-vs-rest
                metrics["auc_roc"] = float(roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average=average
                ))
        except ValueError as e:
            logger.warning(f"Could not compute AUC-ROC: {e}")
            metrics["auc_roc"] = None
    else:
        metrics["auc_roc"] = None
    
    return metrics


def compute_fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    fold_idx: Optional[int] = None,
    class_labels: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, int]]:
    """
    Compute metrics for a single fold with additional metadata.
    
    Args:
        y_true: True labels for this fold
        y_pred: Predicted labels for this fold
        y_prob: Predicted probabilities (optional)
        fold_idx: Fold number (for logging)
        class_labels: Array of class labels from classifier.classes_
        
    Returns:
        Dictionary with metrics and fold metadata
    """
    # Determine if binary or multiclass
    unique_labels = np.unique(y_true)
    is_binary = len(unique_labels) == 2
    
    average = "binary" if is_binary else "macro"
    pos_label = 1 if is_binary else None
    
    # Compute metrics
    metrics = compute_classification_metrics(
        y_true, y_pred, y_prob, average=average, pos_label=pos_label, class_labels=class_labels
    )
    
    # Add fold metadata
    if fold_idx is not None:
        metrics["fold"] = fold_idx
    
    metrics["n_samples"] = len(y_true)
    metrics["n_positive"] = int(np.sum(y_true == 1)) if is_binary else None
    metrics["n_negative"] = int(np.sum(y_true == 0)) if is_binary else None
    
    # Log fold results
    if fold_idx is not None:
        logger.info(f"Fold {fold_idx} metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        if metrics.get("specificity") is not None:
            logger.info(f"  Specificity: {metrics['specificity']:.4f}")
        if metrics.get("auc_roc") is not None:
            logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return metrics


def aggregate_cv_metrics(fold_metrics: list[Dict]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across multiple CV folds.
    
    Args:
        fold_metrics: List of metric dictionaries from each fold
        
    Returns:
        Dictionary with 'mean' and 'std' subdictionaries containing aggregated metrics
    """
    import pandas as pd
    
    # Convert to DataFrame for easy aggregation
    df = pd.DataFrame(fold_metrics)
    
    # Metrics to aggregate (exclude non-numeric and metadata)
    exclude_cols = ["fold", "confusion_matrix", "n_samples", "n_positive", "n_negative"]
    numeric_cols = [col for col in df.columns 
                   if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    # Compute mean and std
    means = df[numeric_cols].mean().to_dict()
    stds = df[numeric_cols].std().to_dict()
    
    # Log summary
    logger.info("Cross-validation results (mean ± std):")
    for metric in ["accuracy", "precision", "recall", "f1", "specificity", "auc_roc"]:
        if metric in means and means[metric] is not None:
            mean_val = means[metric]
            std_val = stds.get(metric, 0.0)
            logger.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    return {
        "mean": means,
        "std": stds,
        "folds": fold_metrics
    }


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list[str]] = None
) -> str:
    """
    Generate a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for each class
        
    Returns:
        Classification report as string
    """
    report = classification_report(y_true, y_pred, target_names=target_names)
    logger.info("Classification Report:\n" + report)
    return report


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix with optional normalization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        
    Returns:
        Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    return cm
