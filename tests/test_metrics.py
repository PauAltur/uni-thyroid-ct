"""
Tests for metrics module.

Tests metrics computation and aggregation functionality.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics import (
    compute_classification_metrics,
    aggregate_fold_metrics,
    log_metrics,
    log_aggregated_metrics,
)


def test_compute_classification_metrics_basic():
    """Test basic metrics computation."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    
    metrics = compute_classification_metrics(y_true, y_pred)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'specificity' in metrics
    assert 'confusion_matrix' in metrics
    
    # Check values
    assert metrics['accuracy'] == 0.875  # 7/8 correct
    assert metrics['true_positives'] == 3
    assert metrics['false_negatives'] == 1
    assert metrics['true_negatives'] == 4
    assert metrics['false_positives'] == 0


def test_compute_classification_metrics_perfect():
    """Test metrics with perfect predictions."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    
    metrics = compute_classification_metrics(y_true, y_pred)
    
    assert metrics['accuracy'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['specificity'] == 1.0


def test_compute_classification_metrics_with_probabilities():
    """Test metrics computation with probabilities for AUC."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.4, 0.7, 0.2])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    
    assert 'auc_roc' in metrics
    assert metrics['auc_roc'] is not None
    assert 0 <= metrics['auc_roc'] <= 1


def test_compute_classification_metrics_all_zeros():
    """Test metrics when all predictions are 0."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    
    metrics = compute_classification_metrics(y_true, y_pred)
    
    assert metrics['accuracy'] == 0.5
    assert metrics['recall'] == 0.0  # No true positives
    assert metrics['specificity'] == 1.0  # All negatives correct


def test_compute_classification_metrics_all_ones():
    """Test metrics when all predictions are 1."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    
    metrics = compute_classification_metrics(y_true, y_pred)
    
    assert metrics['accuracy'] == 0.5
    assert metrics['recall'] == 1.0  # All positives correct
    assert metrics['precision'] == 0.5  # 2 TP, 2 FP


def test_compute_classification_metrics_confusion_matrix():
    """Test confusion matrix computation."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1])
    
    metrics = compute_classification_metrics(y_true, y_pred)
    
    cm = metrics['confusion_matrix']
    assert cm == [[2, 1], [1, 2]]  # [[TN, FP], [FN, TP]]
    
    assert metrics['true_negatives'] == 2
    assert metrics['false_positives'] == 1
    assert metrics['false_negatives'] == 1
    assert metrics['true_positives'] == 2


def test_compute_classification_metrics_support():
    """Test support (class counts) in metrics."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 1, 1])
    
    metrics = compute_classification_metrics(y_true, y_pred)
    
    assert 'support' in metrics
    assert metrics['support'][0] == 3  # 3 negative samples
    assert metrics['support'][1] == 4  # 4 positive samples


def test_aggregate_fold_metrics_basic():
    """Test basic aggregation of fold metrics."""
    fold_results = [
        {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.85, 'f1': 0.8, 
         'specificity': 0.7, 'auc_roc': 0.82, 'confusion_matrix': [[10, 2], [3, 15]]},
        {'accuracy': 0.85, 'precision': 0.8, 'recall': 0.9, 'f1': 0.85,
         'specificity': 0.75, 'auc_roc': 0.88, 'confusion_matrix': [[12, 1], [2, 15]]},
        {'accuracy': 0.9, 'precision': 0.85, 'recall': 0.95, 'f1': 0.9,
         'specificity': 0.8, 'auc_roc': 0.92, 'confusion_matrix': [[14, 0], [1, 15]]},
    ]
    
    aggregated = aggregate_fold_metrics(fold_results)
    
    assert 'accuracy_mean' in aggregated
    assert 'accuracy_std' in aggregated
    assert aggregated['accuracy_mean'] == pytest.approx(0.85, abs=0.01)
    assert aggregated['precision_mean'] == pytest.approx(0.8, abs=0.01)
    assert aggregated['recall_mean'] == pytest.approx(0.9, abs=0.01)


def test_aggregate_fold_metrics_confusion_matrix():
    """Test aggregation of confusion matrices."""
    fold_results = [
        {'confusion_matrix': [[10, 2], [3, 15]]},
        {'confusion_matrix': [[12, 1], [2, 15]]},
        {'confusion_matrix': [[14, 0], [1, 15]]},
    ]
    
    aggregated = aggregate_fold_metrics(fold_results)
    
    assert 'confusion_matrix_total' in aggregated
    total_cm = np.array(aggregated['confusion_matrix_total'])
    assert total_cm.shape == (2, 2)
    assert total_cm.tolist() == [[36, 3], [6, 45]]


def test_aggregate_fold_metrics_with_none_values():
    """Test aggregation when some folds have None values."""
    fold_results = [
        {'accuracy': 0.8, 'auc_roc': 0.85, 'confusion_matrix': [[8, 2], [1, 9]]},
        {'accuracy': 0.85, 'auc_roc': None, 'confusion_matrix': [[9, 1], [2, 8]]},  # AUC not available
        {'accuracy': 0.9, 'auc_roc': 0.9, 'confusion_matrix': [[7, 3], [1, 9]]},
    ]
    
    aggregated = aggregate_fold_metrics(fold_results)
    
    # Accuracy should aggregate all 3 values
    assert aggregated['accuracy_mean'] == pytest.approx(0.85, abs=0.01)
    
    # AUC should only aggregate 2 values (skip None)
    assert aggregated['auc_roc_mean'] == pytest.approx(0.875, abs=0.01)
    
    # Confusion matrix should be aggregated
    assert 'confusion_matrix_total' in aggregated


def test_aggregate_fold_metrics_empty_list():
    """Test aggregation with empty fold results."""
    fold_results = []
    
    aggregated = aggregate_fold_metrics(fold_results)
    
    # Should have None values for all metrics
    assert aggregated['accuracy_mean'] is None
    assert aggregated['accuracy_std'] is None


def test_aggregate_fold_metrics_std_calculation():
    """Test standard deviation calculation."""
    fold_results = [
        {'accuracy': 0.6, 'precision': 0.5, 'recall': 0.7, 'f1': 0.6,
         'specificity': 0.5, 'auc_roc': 0.65, 'confusion_matrix': [[10, 10], [5, 15]]},
        {'accuracy': 0.8, 'precision': 0.7, 'recall': 0.9, 'f1': 0.8,
         'specificity': 0.7, 'auc_roc': 0.85, 'confusion_matrix': [[14, 6], [2, 18]]},
        {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
         'specificity': 1.0, 'auc_roc': 1.0, 'confusion_matrix': [[20, 0], [0, 20]]},
    ]
    
    aggregated = aggregate_fold_metrics(fold_results)
    
    # Mean should be 0.8
    assert aggregated['accuracy_mean'] == pytest.approx(0.8, abs=0.01)
    # Std should be > 0 (values are 0.6, 0.8, 1.0)
    assert aggregated['accuracy_std'] > 0


def test_log_metrics(caplog):
    """Test logging of metrics."""
    import logging
    caplog.set_level(logging.INFO)
    
    metrics = {
        'accuracy': 0.85,
        'precision': 0.80,
        'recall': 0.90,
        'specificity': 0.75,
        'f1': 0.85,
        'auc_roc': 0.88,
    }
    
    log_metrics(metrics, prefix="Test Fold:")
    
    # Check that metrics were logged
    assert "Test Fold:" in caplog.text
    assert "0.85" in caplog.text or "0.8500" in caplog.text


def test_log_aggregated_metrics(caplog):
    """Test logging of aggregated metrics."""
    import logging
    caplog.set_level(logging.INFO)
    
    aggregated = {
        'accuracy_mean': 0.85,
        'accuracy_std': 0.05,
        'precision_mean': 0.80,
        'precision_std': 0.06,
        'recall_mean': 0.90,
        'recall_std': 0.04,
    }
    
    log_aggregated_metrics(aggregated)
    
    # Check that aggregated results were logged
    assert "Aggregated Results" in caplog.text
    assert "Mean ± Std" in caplog.text
