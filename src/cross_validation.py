"""
Cross-validation utilities for group-aware splitting.

This module provides functions for k-fold cross-validation that ensures
samples (groups) are not split across folds, preventing data leakage.
"""

import numpy as np
from sklearn.model_selection import GroupKFold
from typing import Tuple, List, Generator
import logging

logger = logging.getLogger(__name__)


def create_group_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Create k-fold cross-validation splits with group-aware splitting.
    
    Ensures that all embeddings from the same sample (group) stay in the same fold,
    preventing data leakage when samples have multiple patches.
    
    Args:
        X: Feature matrix (N, features)
        y: Target labels (N,)
        groups: Sample identifiers (N,) - embeddings with same group stay together
        n_folds: Number of folds for cross-validation
        shuffle: Whether to shuffle groups before splitting
        random_state: Random seed for reproducibility
        
    Yields:
        train_idx: Indices for training set
        test_idx: Indices for test/validation set
        
    Example:
        >>> for fold, (train_idx, test_idx) in enumerate(create_group_kfold_splits(X, y, groups)):
        >>>     X_train, X_test = X[train_idx], X[test_idx]
        >>>     y_train, y_test = y[train_idx], y[test_idx]
        >>>     # Train and evaluate model
    """
    # Validate inputs
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(
            f"X ({len(X)}), y ({len(y)}), and groups ({len(groups)}) "
            f"must have the same length"
        )
    
    unique_groups = np.unique(groups)
    
    if len(unique_groups) < n_folds:
        raise ValueError(
            f"Number of unique groups ({len(unique_groups)}) is less than "
            f"n_folds ({n_folds}). Cannot create {n_folds} folds."
        )
    
    # Warn if folds will be very small
    avg_groups_per_fold = len(unique_groups) / n_folds
    if avg_groups_per_fold < 3:
        logger.warning(
            f"Small number of groups per fold (avg {avg_groups_per_fold:.1f}). "
            f"Consider reducing n_folds or collecting more data. "
            f"Metrics may be unreliable with very small test sets."
        )
    
    logger.info(f"Creating {n_folds}-fold cross-validation splits")
    logger.info(f"  Total samples: {len(X)}")
    logger.info(f"  Unique groups: {len(unique_groups)}")
    logger.info(f"  Samples per group (mean): {len(X) / len(unique_groups):.1f}")
    
    # Create GroupKFold splitter
    groups_to_use = groups.copy()
    
    if shuffle:
        # Shuffle by creating a randomized mapping of original groups to new group IDs
        # This ensures GroupKFold splits shuffled groups while maintaining group integrity
        np.random.seed(random_state)
        
        # Create mapping: original_group -> shuffled_integer_id
        unique_groups_list = list(unique_groups)
        shuffled_indices = np.random.permutation(len(unique_groups_list))
        group_to_id = {unique_groups_list[i]: shuffled_indices[i] for i in range(len(unique_groups_list))}
        
        # Apply mapping to all samples
        groups_to_use = np.array([group_to_id[g] for g in groups])
        logger.info(f"  Groups shuffled with random_state={random_state}")
    
    splitter = GroupKFold(n_splits=n_folds)
    split_iterator = splitter.split(X, y, groups=groups_to_use)
    
    # Generate splits and log statistics
    for fold_idx, (train_idx, test_idx) in enumerate(split_iterator, 1):
        # Get groups in each split
        train_groups = np.unique(groups[train_idx])
        test_groups = np.unique(groups[test_idx])
        
        # Validate no overlap
        overlap = set(train_groups) & set(test_groups)
        if overlap:
            logger.warning(f"Fold {fold_idx}: Found overlapping groups: {overlap}")
        
        # Log fold statistics
        logger.info(f"Fold {fold_idx}/{n_folds}:")
        logger.info(f"  Train: {len(train_idx)} samples, {len(train_groups)} groups")
        logger.info(f"  Test:  {len(test_idx)} samples, {len(test_groups)} groups")
        
        # Warn if test set is very small
        if len(test_idx) < 10:
            logger.warning(
                f"Very small test set in fold {fold_idx}: only {len(test_idx)} samples. "
                f"Metrics may be unreliable."
            )
        
        # Check label distribution
        train_dist = np.bincount(y[train_idx])
        test_dist = np.bincount(y[test_idx])
        logger.info(f"  Train label dist: {dict(enumerate(train_dist))}")
        logger.info(f"  Test label dist:  {dict(enumerate(test_dist))}")
        
        yield train_idx, test_idx


def validate_group_split(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    groups: np.ndarray
) -> bool:
    """
    Validate that there's no group leakage between train and test sets.
    
    Args:
        train_idx: Indices of training samples
        test_idx: Indices of test samples
        groups: Sample group identifiers
        
    Returns:
        True if split is valid (no overlap), False otherwise
    """
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    
    overlap = train_groups & test_groups
    
    if overlap:
        logger.error(f"Found {len(overlap)} overlapping groups between train and test!")
        logger.error(f"Overlapping groups: {overlap}")
        return False
    
    return True


def stratified_group_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Create stratified k-fold splits while respecting group structure.
    
    This attempts to balance class distribution across folds while ensuring
    groups stay together. Note: Perfect stratification may not be possible
    when groups have imbalanced labels.
    
    Args:
        X: Feature matrix (N, features)
        y: Target labels (N,)
        groups: Sample identifiers (N,)
        n_folds: Number of folds
        random_state: Random seed
        
    Yields:
        train_idx: Training indices
        test_idx: Test indices
    """
    # Calculate label distribution per group
    unique_groups = np.unique(groups)
    group_labels = {}
    
    for group in unique_groups:
        group_mask = groups == group
        # Use majority label for the group
        group_label = np.bincount(y[group_mask]).argmax()
        group_labels[group] = group_label
    
    # Convert to arrays for sklearn
    group_label_array = np.array([group_labels[g] for g in unique_groups])
    
    # Sort groups by label to improve stratification
    sorted_indices = np.argsort(group_label_array)
    sorted_groups = unique_groups[sorted_indices]
    
    # Distribute groups across folds in round-robin fashion
    fold_groups = [[] for _ in range(n_folds)]
    
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(sorted_groups))
    
    for i, idx in enumerate(shuffled_indices):
        fold_idx = i % n_folds
        fold_groups[fold_idx].append(sorted_groups[idx])
    
    # Generate splits
    for fold_idx in range(n_folds):
        test_groups_fold = fold_groups[fold_idx]
        train_groups_fold = [g for i, groups_list in enumerate(fold_groups) 
                            if i != fold_idx for g in groups_list]
        
        # Get indices
        test_mask = np.isin(groups, test_groups_fold)
        train_mask = np.isin(groups, train_groups_fold)
        
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        
        logger.info(f"Stratified Fold {fold_idx + 1}/{n_folds}:")
        logger.info(f"  Train: {len(train_idx)} samples, {len(train_groups_fold)} groups")
        logger.info(f"  Test:  {len(test_idx)} samples, {len(test_groups_fold)} groups")
        
        yield train_idx, test_idx
