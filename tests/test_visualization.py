"""
Tests for visualization module.

Tests plotting and visualization functionality.
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization import (
    plot_cv_results,
    plot_roc_curves,
    create_results_summary_table,
)


@pytest.fixture
def sample_fold_results():
    """Create sample fold results for testing."""
    fold_results = []
    for i in range(5):
        fold_results.append({
            'fold': i + 1,
            'train_size': 80,
            'test_size': 20,
            'train_samples': 8,
            'test_samples': 2,
            'accuracy': 0.8 + i * 0.02,
            'precision': 0.75 + i * 0.03,
            'recall': 0.85 + i * 0.01,
            'specificity': 0.7 + i * 0.04,
            'f1': 0.8 + i * 0.02,
            'auc_roc': 0.82 + i * 0.02,
            'confusion_matrix': [[8, 2], [1, 9]],
            'true_positives': 9,
            'false_positives': 2,
            'true_negatives': 8,
            'false_negatives': 1,
        })
    return fold_results


@pytest.fixture
def sample_aggregated_metrics():
    """Create sample aggregated metrics for testing."""
    return {
        'accuracy_mean': 0.85,
        'accuracy_std': 0.05,
        'precision_mean': 0.80,
        'precision_std': 0.06,
        'recall_mean': 0.90,
        'recall_std': 0.04,
        'specificity_mean': 0.75,
        'specificity_std': 0.05,
        'f1_mean': 0.85,
        'f1_std': 0.04,
        'auc_roc_mean': 0.88,
        'auc_roc_std': 0.03,
        'confusion_matrix_total': [[40, 10], [5, 45]],
    }


def test_plot_cv_results_creates_file(sample_fold_results, sample_aggregated_metrics):
    """Test that plot file is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plot.png"
        
        plot_cv_results(
            sample_fold_results,
            sample_aggregated_metrics,
            output_path,
            n_folds=5
        )
        
        assert output_path.exists()


def test_plot_cv_results_creates_directory(sample_fold_results, sample_aggregated_metrics):
    """Test that output directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "subdir" / "test_plot.png"
        
        plot_cv_results(
            sample_fold_results,
            sample_aggregated_metrics,
            output_path,
            n_folds=5
        )
        
        assert output_path.exists()
        assert output_path.parent.exists()


def test_plot_cv_results_with_missing_metrics(sample_aggregated_metrics):
    """Test plotting with some missing metrics."""
    fold_results = [
        {
            'fold': 1,
            'train_size': 80,
            'test_size': 20,
            'train_samples': 8,
            'test_samples': 2,
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.85,
            'specificity': 0.75,
            'f1': 0.8,
            'auc_roc': None,  # Missing
            'confusion_matrix': [[8, 2], [1, 9]],
            'true_positives': 9,
            'false_positives': 2,
            'true_negatives': 8,
            'false_negatives': 1,
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plot.png"
        
        # Should not raise an error
        plot_cv_results(
            fold_results,
            sample_aggregated_metrics,
            output_path,
            n_folds=1
        )
        
        assert output_path.exists()


def test_plot_roc_curves_creates_file():
    """Test that ROC curve plot is created."""
    # Create sample data
    fold_metadata = []
    y_true_list = []
    
    for i in range(3):
        fold_metadata.append({
            'fold': i + 1,
            'y_proba': np.random.uniform(0, 1, 20),
        })
        y_true_list.append(np.random.randint(0, 2, 20))
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "roc_curves.png"
        
        plot_roc_curves(fold_metadata, y_true_list, output_path)
        
        assert output_path.exists()


def test_plot_roc_curves_handles_missing_probabilities():
    """Test ROC curve plotting when some folds don't have probabilities."""
    fold_metadata = [
        {'fold': 1, 'y_proba': np.random.uniform(0, 1, 20)},
        {'fold': 2, 'y_proba': None},  # No probabilities
        {'fold': 3, 'y_proba': np.random.uniform(0, 1, 20)},
    ]
    y_true_list = [
        np.random.randint(0, 2, 20),
        np.random.randint(0, 2, 20),
        np.random.randint(0, 2, 20),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "roc_curves.png"
        
        # Should complete without error
        plot_roc_curves(fold_metadata, y_true_list, output_path)
        
        assert output_path.exists()


def test_create_results_summary_table(sample_fold_results, sample_aggregated_metrics):
    """Test creating results summary table."""
    summary = create_results_summary_table(
        sample_fold_results,
        sample_aggregated_metrics
    )
    
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 6  # 5 folds + 1 aggregated row
    assert 'fold' in summary.columns
    
    # Last row should be aggregated
    assert summary.iloc[-1]['fold'] == "Mean ± Std"


def test_create_results_summary_table_with_none_metrics():
    """Test summary table with None metrics."""
    fold_results = [
        {'fold': 1, 'accuracy': 0.8, 'auc_roc': None},
        {'fold': 2, 'accuracy': 0.85, 'auc_roc': None},
    ]
    aggregated_metrics = {
        'accuracy_mean': 0.825,
        'accuracy_std': 0.025,
        'auc_roc_mean': None,
        'auc_roc_std': None,
    }
    
    summary = create_results_summary_table(fold_results, aggregated_metrics)
    
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 3  # 2 folds + 1 aggregated


def test_plot_cv_results_different_n_folds(sample_aggregated_metrics):
    """Test plotting with different numbers of folds."""
    for n_folds in [3, 5, 10]:
        fold_results = [
            {
                'fold': i + 1,
                'accuracy': 0.8,
                'precision': 0.75,
                'recall': 0.85,
                'f1': 0.8,
                'confusion_matrix': [[8, 2], [1, 9]],
            }
            for i in range(n_folds)
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / f"plot_{n_folds}folds.png"
            
            plot_cv_results(
                fold_results,
                sample_aggregated_metrics,
                output_path,
                n_folds=n_folds
            )
            
            assert output_path.exists()


def test_plot_cv_results_file_is_valid_image(sample_fold_results, sample_aggregated_metrics):
    """Test that generated plot is a valid PNG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plot.png"
        
        plot_cv_results(
            sample_fold_results,
            sample_aggregated_metrics,
            output_path,
            n_folds=5
        )
        
        # Check file exists and is not empty
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Check it starts with PNG magic number
        with open(output_path, 'rb') as f:
            header = f.read(8)
            # PNG files start with these bytes
            assert header[:4] == b'\x89PNG'
