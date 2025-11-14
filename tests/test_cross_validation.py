"""
Tests for cross_validation module.

Tests K-fold cross-validation with sample-level splitting.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cross_validation import (
    get_group_kfold_splits,
    train_and_evaluate_fold,
    run_cross_validation,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    # Create 10 groups with 10 samples each
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    groups = np.array([f"G{i//10}" for i in range(n_samples)])
    
    return X, y, groups


def test_get_group_kfold_splits_basic(sample_data):
    """Test basic K-fold split generation."""
    X, y, groups = sample_data
    
    splits = get_group_kfold_splits(X, y, groups, n_folds=5)
    
    assert len(splits) == 5
    
    # Check each split
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert len(train_idx) + len(test_idx) == len(X)
        assert len(set(train_idx) & set(test_idx)) == 0  # No overlap


def test_get_group_kfold_splits_no_leakage(sample_data):
    """Test that samples from same group don't leak between folds."""
    X, y, groups = sample_data
    
    splits = get_group_kfold_splits(X, y, groups, n_folds=5)
    
    for train_idx, test_idx in splits:
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        
        # No group should appear in both train and test
        assert len(train_groups & test_groups) == 0


def test_get_group_kfold_splits_coverage(sample_data):
    """Test that all samples are used exactly once in test sets."""
    X, y, groups = sample_data
    
    splits = get_group_kfold_splits(X, y, groups, n_folds=5)
    
    # Collect all test indices
    all_test_indices = set()
    for train_idx, test_idx in splits:
        all_test_indices.update(test_idx)
    
    # All indices should appear exactly once
    assert all_test_indices == set(range(len(X)))


def test_train_and_evaluate_fold_basic(sample_data):
    """Test training and evaluating a single fold."""
    X, y, groups = sample_data
    
    # Create a simple split
    n = len(X)
    train_idx = np.arange(0, n * 4 // 5)
    test_idx = np.arange(n * 4 // 5, n)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]
    
    clf = LogisticRegression(random_state=42)
    
    result = train_and_evaluate_fold(
        clf, X_train, y_train, X_test, y_test,
        groups_train, groups_test, fold_idx=1, scale_features=True
    )
    
    assert 'y_pred' in result
    assert 'y_proba' in result
    assert 'train_size' in result
    assert 'test_size' in result
    assert 'train_samples' in result
    assert 'test_samples' in result
    
    assert len(result['y_pred']) == len(y_test)
    assert result['train_size'] == len(X_train)
    assert result['test_size'] == len(X_test)


def test_train_and_evaluate_fold_with_scaling(sample_data):
    """Test that feature scaling is applied when requested."""
    X, y, groups = sample_data
    
    n = len(X)
    train_idx = np.arange(0, n * 4 // 5)
    test_idx = np.arange(n * 4 // 5, n)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]
    
    clf = LogisticRegression(random_state=42)
    
    # With scaling
    result_scaled = train_and_evaluate_fold(
        clf, X_train, y_train, X_test, y_test,
        groups_train, groups_test, fold_idx=1, scale_features=True
    )
    
    # Without scaling
    clf2 = LogisticRegression(random_state=42)
    result_unscaled = train_and_evaluate_fold(
        clf2, X_train, y_train, X_test, y_test,
        groups_train, groups_test, fold_idx=1, scale_features=False
    )
    
    # Both should produce predictions
    assert len(result_scaled['y_pred']) == len(y_test)
    assert len(result_unscaled['y_pred']) == len(y_test)


def test_train_and_evaluate_fold_detects_leakage():
    """Test that sample leakage is detected."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Create overlapping groups (leakage)
    groups_train = np.array(['A'] * 50 + ['B'] * 30)
    groups_test = np.array(['B'] * 20)  # B appears in both!
    
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    clf = LogisticRegression(random_state=42)
    
    with pytest.raises(ValueError, match="Sample leakage detected"):
        train_and_evaluate_fold(
            clf, X_train, y_train, X_test, y_test,
            groups_train, groups_test, fold_idx=1
        )


def test_run_cross_validation_basic(sample_data):
    """Test complete cross-validation run."""
    X, y, groups = sample_data
    clf = LogisticRegression(random_state=42)
    
    fold_metadata, y_true_list, y_pred_list = run_cross_validation(
        X, y, groups, clf, n_folds=5, scale_features=True
    )
    
    assert len(fold_metadata) == 5
    assert len(y_true_list) == 5
    assert len(y_pred_list) == 5
    
    # Check each fold
    for i in range(5):
        assert len(y_true_list[i]) == len(y_pred_list[i])
        assert len(y_true_list[i]) > 0
        
        # Check metadata
        assert 'fold' in fold_metadata[i]
        assert fold_metadata[i]['fold'] == i + 1
        assert 'train_size' in fold_metadata[i]
        assert 'test_size' in fold_metadata[i]
        assert 'y_proba' in fold_metadata[i]


def test_run_cross_validation_predictions_valid(sample_data):
    """Test that predictions are valid."""
    X, y, groups = sample_data
    clf = LogisticRegression(random_state=42)
    
    fold_metadata, y_true_list, y_pred_list = run_cross_validation(
        X, y, groups, clf, n_folds=5
    )
    
    for y_pred in y_pred_list:
        # Predictions should be binary
        assert set(np.unique(y_pred)).issubset({0, 1})


def test_run_cross_validation_probabilities(sample_data):
    """Test that probabilities are returned when available."""
    X, y, groups = sample_data
    clf = LogisticRegression(random_state=42)  # Supports predict_proba
    
    fold_metadata, y_true_list, y_pred_list = run_cross_validation(
        X, y, groups, clf, n_folds=5
    )
    
    for fold_meta in fold_metadata:
        assert fold_meta['y_proba'] is not None
        # Probabilities should be between 0 and 1
        assert np.all(fold_meta['y_proba'] >= 0)
        assert np.all(fold_meta['y_proba'] <= 1)


def test_run_cross_validation_all_samples_tested(sample_data):
    """Test that all samples are tested exactly once."""
    X, y, groups = sample_data
    clf = LogisticRegression(random_state=42)
    
    fold_metadata, y_true_list, y_pred_list = run_cross_validation(
        X, y, groups, clf, n_folds=5
    )
    
    # Collect all test predictions
    total_tested = sum(len(y_true) for y_true in y_true_list)
    
    # Should equal total samples
    assert total_tested == len(X)


def test_run_cross_validation_no_group_leakage(sample_data):
    """Test that groups don't leak between train and test in CV."""
    X, y, groups = sample_data
    clf = LogisticRegression(random_state=42)
    
    # This should complete without raising ValueError
    # (train_and_evaluate_fold checks for leakage)
    fold_metadata, y_true_list, y_pred_list = run_cross_validation(
        X, y, groups, clf, n_folds=5
    )
    
    # If we got here, no leakage was detected
    assert len(fold_metadata) == 5


def test_run_cross_validation_different_n_folds():
    """Test CV with different numbers of folds."""
    np.random.seed(42)
    X = np.random.randn(120, 10)
    y = np.random.randint(0, 2, 120)
    groups = np.array([f"G{i//10}" for i in range(120)])  # 12 groups
    
    clf = LogisticRegression(random_state=42)
    
    for n_folds in [3, 4, 6]:
        fold_metadata, y_true_list, y_pred_list = run_cross_validation(
            X, y, groups, clf, n_folds=n_folds
        )
        
        assert len(fold_metadata) == n_folds
        assert len(y_true_list) == n_folds
        assert len(y_pred_list) == n_folds
