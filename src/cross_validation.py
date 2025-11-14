"""
Model training and cross-validation utilities.

This module provides functions for training classifiers and performing
K-fold cross-validation with sample-level splitting.
"""

import logging
from typing import Dict, List, Any, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def get_group_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate K-fold cross-validation splits with sample-level grouping.
    
    Ensures all patches from the same sample stay together in the same fold
    to prevent data leakage.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        groups: Group identifiers for each sample (n_samples,)
        n_folds: Number of folds
    
    Returns:
        List of (train_indices, test_indices) tuples for each fold
    
    Example:
        >>> splits = get_group_kfold_splits(X, y, groups, n_folds=5)
        >>> for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        ...     print(f"Fold {fold_idx}: {len(train_idx)} train, {len(test_idx)} test")
    """
    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(X, y, groups))
    
    logger.info(f"Generated {n_folds} folds with sample-level splitting")
    
    return splits


def train_and_evaluate_fold(
    clf: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    groups_test: np.ndarray,
    fold_idx: int,
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Train classifier on one fold and return predictions and metadata.
    
    Args:
        clf: Scikit-learn classifier instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        groups_train: Training sample groups
        groups_test: Test sample groups
        fold_idx: Current fold number (for logging)
        scale_features: Whether to standardize features
    
    Returns:
        Dictionary containing:
            - y_pred: Predicted labels
            - y_proba: Predicted probabilities (if available)
            - train_size: Number of training samples
            - test_size: Number of test samples
            - train_samples: Number of unique training samples
            - test_samples: Number of unique test samples
    
    Example:
        >>> result = train_and_evaluate_fold(
        ...     clf, X_train, y_train, X_test, y_test,
        ...     groups_train, groups_test, fold_idx=1
        ... )
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Fold {fold_idx}")
    logger.info(f"{'='*60}")
    
    # Get unique samples in each split
    train_samples = np.unique(groups_train)
    test_samples = np.unique(groups_test)
    
    logger.info(
        f"Train: {len(X_train):,} patches from {len(train_samples)} samples"
    )
    logger.info(
        f"Test:  {len(X_test):,} patches from {len(test_samples)} samples"
    )
    logger.info(
        f"Train class distribution: "
        f"{dict(zip(*np.unique(y_train, return_counts=True)))}"
    )
    logger.info(
        f"Test class distribution:  "
        f"{dict(zip(*np.unique(y_test, return_counts=True)))}"
    )
    
    # Verify no sample leakage
    overlap = set(train_samples) & set(test_samples)
    if overlap:
        raise ValueError(
            f"Sample leakage detected! {len(overlap)} samples in both "
            f"train and test: {list(overlap)[:5]}"
        )
    
    # Scale features if requested
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Train classifier
    logger.info(f"Training {clf.__class__.__name__}...")
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Get predicted probabilities if available
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = None
        logger.warning("Classifier does not support predict_proba, AUC-ROC will not be computed")
    
    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
    }


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf: BaseEstimator,
    n_folds: int,
    scale_features: bool = True,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[np.ndarray]]:
    """
    Run K-fold cross-validation with sample-level splitting.
    
    This is the main function that orchestrates the entire cross-validation process.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        groups: Group identifiers (n_samples,)
        clf: Scikit-learn classifier instance (will be cloned for each fold)
        n_folds: Number of folds
        scale_features: Whether to standardize features
    
    Returns:
        Tuple of:
            - fold_metadata: List of dictionaries with fold information
            - y_true_list: List of true labels for each fold
            - y_pred_list: List of predicted labels for each fold
    
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> clf = LogisticRegression()
        >>> fold_meta, y_true_folds, y_pred_folds = run_cross_validation(
        ...     X, y, groups, clf, n_folds=5
        ... )
    """
    logger.info(f"Starting {n_folds}-fold cross-validation")
    logger.info(f"Total patches: {len(X):,}")
    logger.info(f"Unique samples: {len(np.unique(groups))}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Get fold splits
    splits = get_group_kfold_splits(X, y, groups, n_folds)
    
    fold_metadata = []
    y_true_list = []
    y_pred_list = []
    
    # Iterate through folds
    for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx]
        groups_test = groups[test_idx]
        
        # Clone classifier for this fold
        from sklearn.base import clone
        clf_fold = clone(clf)
        
        # Train and evaluate
        fold_result = train_and_evaluate_fold(
            clf_fold,
            X_train,
            y_train,
            X_test,
            y_test,
            groups_train,
            groups_test,
            fold_idx,
            scale_features,
        )
        
        # Store metadata
        fold_metadata.append({
            "fold": fold_idx,
            "train_size": fold_result["train_size"],
            "test_size": fold_result["test_size"],
            "train_samples": fold_result["train_samples"],
            "test_samples": fold_result["test_samples"],
            "y_proba": fold_result["y_proba"],
        })
        
        y_true_list.append(y_test)
        y_pred_list.append(fold_result["y_pred"])
    
    logger.info(f"\n{'='*60}")
    logger.info("Cross-validation completed")
    logger.info(f"{'='*60}")
    
    return fold_metadata, y_true_list, y_pred_list
