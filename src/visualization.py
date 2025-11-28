"""
Visualization utilities for classification results.

This module provides functions to create plots and visualizations
for cross-validation results.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_cv_results(
    fold_results: List[Dict[str, Any]],
    aggregated_metrics: Dict[str, Any],
    output_path: Path,
    n_folds: int,
) -> None:
    """
    Generate comprehensive visualization of cross-validation results.
    
    Creates a 4-panel figure with:
        1. Metrics across folds (bar plot)
        2. Aggregated metrics with error bars
        3. Total confusion matrix (heatmap)
        4. Metric distribution across folds (box plot)
    
    Args:
        fold_results: List of dictionaries with fold-level metrics
        aggregated_metrics: Dictionary with aggregated metrics
        output_path: Path to save the plot
        n_folds: Number of folds
    
    Example:
        >>> plot_cv_results(fold_results, aggregated_metrics, 
        ...                 Path("results/plots.png"), n_folds=5)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating visualization with {n_folds} folds")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Metrics across folds (bar plot)
    _plot_metrics_across_folds(axes[0, 0], fold_results, n_folds)
    
    # 2. Aggregated metrics with error bars
    _plot_aggregated_metrics(axes[0, 1], aggregated_metrics)
    
    # 3. Total confusion matrix
    _plot_confusion_matrix(axes[1, 0], aggregated_metrics)
    
    # 4. Fold-wise performance comparison (box plot)
    _plot_metric_distribution(axes[1, 1], fold_results)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plots to {output_path}")
    plt.close()


def _plot_metrics_across_folds(ax, fold_results: List[Dict], n_folds: int) -> None:
    """Plot metrics for each fold as grouped bars."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    fold_numbers = [f"Fold {i}" for i in range(1, n_folds + 1)]
    
    x = np.arange(len(fold_numbers))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [fold.get(metric, 0) for fold in fold_results]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics Across Folds")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(fold_numbers)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)


def _plot_aggregated_metrics(ax, aggregated_metrics: Dict) -> None:
    """Plot aggregated metrics with error bars."""
    metrics_for_agg = ["accuracy", "precision", "recall", "specificity", "f1"]
    
    # Filter out None values
    means = []
    stds = []
    metric_labels = []
    
    for m in metrics_for_agg:
        mean = aggregated_metrics.get(f"{m}_mean")
        std = aggregated_metrics.get(f"{m}_std")
        if mean is not None:
            means.append(mean)
            stds.append(std)
            metric_labels.append(m.capitalize())
    
    if not means:
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        ax.set_title("Aggregated Metrics (Mean ± Std)")
        return
    
    x = np.arange(len(metric_labels))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.set_ylabel("Score")
    ax.set_title("Aggregated Metrics (Mean ± Std)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)


def _plot_confusion_matrix(ax, aggregated_metrics: Dict) -> None:
    """Plot total confusion matrix as heatmap."""
    cm = np.array(aggregated_metrics.get("confusion_matrix_total", [[0, 0], [0, 0]]))
    
    if cm.shape != (2, 2):
        ax.text(0.5, 0.5, "Confusion matrix not available", ha="center", va="center")
        ax.set_title("Total Confusion Matrix (All Folds)")
        return
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Total Confusion Matrix (All Folds)")
    ax.set_xticklabels(['No Invasion', 'Invasion'])
    ax.set_yticklabels(['No Invasion', 'Invasion'])


def _plot_metric_distribution(ax, fold_results: List[Dict]) -> None:
    """Plot distribution of metrics across folds as box plot."""
    data_for_box = []
    labels_for_box = []
    
    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = [
            fold[metric] 
            for fold in fold_results 
            if fold.get(metric) is not None
        ]
        if values:
            data_for_box.append(values)
            labels_for_box.append(metric.capitalize())
    
    if not data_for_box:
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        ax.set_title("Metric Distribution Across Folds")
        return
    
    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_ylabel("Score")
    ax.set_title("Metric Distribution Across Folds")
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)


def plot_roc_curves(
    fold_results: List[Dict[str, Any]],
    y_true_list: List[np.ndarray],
    output_path: Path,
) -> None:
    """
    Plot ROC curves for each fold (if probabilities available).
    
    Args:
        fold_results: List of dictionaries with fold metadata including y_proba
        y_true_list: List of true labels for each fold
        output_path: Path to save the plot
    
    Example:
        >>> plot_roc_curves(fold_results, y_true_list, Path("results/roc_curves.png"))
    """
    from sklearn.metrics import roc_curve, auc
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve for each fold
    for fold_idx, (fold_meta, y_true) in enumerate(zip(fold_results, y_true_list), 1):
        y_proba = fold_meta.get("y_proba")
        
        if y_proba is None:
            logger.warning(f"Fold {fold_idx}: No probabilities available, skipping ROC curve")
            continue
        
        try:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=2, alpha=0.7, 
                   label=f'Fold {fold_idx} (AUC = {roc_auc:.3f})')
        except Exception as e:
            logger.warning(f"Fold {fold_idx}: Could not plot ROC curve: {e}")
            continue
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves for Each Fold', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved ROC curves to {output_path}")
    plt.close()


def create_results_summary_table(
    fold_results: List[Dict[str, Any]],
    aggregated_metrics: Dict[str, Any],
) -> pd.DataFrame:
    """
    Create a formatted summary table of results.
    
    Args:
        fold_results: List of dictionaries with fold-level metrics
        aggregated_metrics: Dictionary with aggregated metrics
    
    Returns:
        DataFrame with formatted results
    
    Example:
        >>> summary = create_results_summary_table(fold_results, aggregated_metrics)
        >>> print(summary.to_string())
    """
    # Create DataFrame from fold results
    df = pd.DataFrame(fold_results)
    
    # Add aggregated row
    agg_row = {"fold": "Mean ± Std"}
    
    for metric in ["accuracy", "precision", "recall", "specificity", "f1", "auc_roc"]:
        mean = aggregated_metrics.get(f"{metric}_mean")
        std = aggregated_metrics.get(f"{metric}_std")
        
        if mean is not None:
            agg_row[metric] = f"{mean:.3f} ± {std:.3f}"
        else:
            agg_row[metric] = "N/A"
    
    # Append aggregated row
    df_summary = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)
    
    return df_summary
