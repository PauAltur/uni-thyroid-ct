"""
Sample-level aggregation utilities.

This module provides functions to aggregate patch-level embeddings and predictions
to sample-level for more robust classification.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def aggregate_embeddings_by_sample(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    sample_column: str = "sample_codes",
    label_column: str = "has_invasion",
    aggregation_method: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate patch-level embeddings to sample-level.
    
    Args:
        embeddings: Patch-level embeddings (N_patches, embedding_dim)
        metadata: Metadata DataFrame with sample IDs and labels
        sample_column: Column containing sample identifiers
        label_column: Column containing labels
        aggregation_method: How to aggregate ('mean', 'max', 'median')
        
    Returns:
        sample_embeddings: Aggregated embeddings (N_samples, embedding_dim)
        sample_labels: Sample-level labels (N_samples,)
        sample_ids: Sample identifiers (N_samples,)
    """
    unique_samples = metadata[sample_column].unique()
    n_samples = len(unique_samples)
    embedding_dim = embeddings.shape[1]
    
    sample_embeddings = np.zeros((n_samples, embedding_dim))
    sample_labels = np.zeros(n_samples, dtype=int)
    sample_ids = []
    
    logger.info(f"Aggregating {len(embeddings)} patches to {n_samples} samples using '{aggregation_method}'")
    
    for i, sample_id in enumerate(unique_samples):
        # Get all patches for this sample
        mask = metadata[sample_column] == sample_id
        sample_patches = embeddings[mask]
        sample_label_values = metadata.loc[mask, label_column].values
        
        # Aggregate embeddings
        if aggregation_method == "mean":
            sample_embeddings[i] = np.mean(sample_patches, axis=0)
        elif aggregation_method == "max":
            sample_embeddings[i] = np.max(sample_patches, axis=0)
        elif aggregation_method == "median":
            sample_embeddings[i] = np.median(sample_patches, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Get sample label (should be consistent across patches)
        unique_labels = np.unique(sample_label_values)
        if len(unique_labels) > 1:
            logger.warning(
                f"Sample {sample_id} has inconsistent labels: {unique_labels}. "
                f"Using majority vote."
            )
            sample_labels[i] = np.bincount(sample_label_values.astype(int)).argmax()
        else:
            sample_labels[i] = unique_labels[0]
        
        sample_ids.append(sample_id)
    
    # Log statistics
    unique_labels = np.unique(sample_labels)
    logger.info(f"Sample-level aggregation complete:")
    logger.info(f"  Total samples: {n_samples}")
    logger.info(f"  Label distribution: {dict(zip(*np.unique(sample_labels, return_counts=True)))}")
    
    label_counts = np.bincount(sample_labels)
    if len(label_counts) == 2:
        ratio = label_counts[0] / label_counts[1] if label_counts[1] > 0 else float('inf')
        logger.info(f"  Class imbalance ratio: {ratio:.2f}:1")
    
    return sample_embeddings, sample_labels, np.array(sample_ids)


def aggregate_patch_predictions_to_sample(
    patch_predictions: np.ndarray,
    patch_probabilities: np.ndarray,
    sample_ids: np.ndarray,
    aggregation_method: str = "mean_prob",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate patch-level predictions to sample-level predictions.
    
    Args:
        patch_predictions: Predicted labels for patches (N_patches,)
        patch_probabilities: Predicted probabilities for patches (N_patches, n_classes)
        sample_ids: Sample identifier for each patch (N_patches,)
        aggregation_method: Aggregation strategy:
            - 'mean_prob': Average probabilities, then threshold
            - 'majority_vote': Most common prediction
            - 'max_prob': Use max probability across patches
        
    Returns:
        sample_predictions: Predictions for samples (N_samples,)
        sample_probabilities: Probabilities for samples (N_samples, n_classes)
        unique_sample_ids: Sample identifiers (N_samples,)
    """
    unique_samples = np.unique(sample_ids)
    n_samples = len(unique_samples)
    n_classes = patch_probabilities.shape[1] if patch_probabilities.ndim > 1 else 2
    
    sample_predictions = np.zeros(n_samples, dtype=int)
    sample_probabilities = np.zeros((n_samples, n_classes))
    
    for i, sample_id in enumerate(unique_samples):
        mask = sample_ids == sample_id
        
        if aggregation_method == "mean_prob":
            # Average probabilities across patches
            if patch_probabilities.ndim > 1:
                sample_probabilities[i] = np.mean(patch_probabilities[mask], axis=0)
            else:
                sample_probabilities[i, 1] = np.mean(patch_probabilities[mask])
                sample_probabilities[i, 0] = 1 - sample_probabilities[i, 1]
            sample_predictions[i] = np.argmax(sample_probabilities[i])
            
        elif aggregation_method == "majority_vote":
            # Most common prediction
            sample_predictions[i] = np.bincount(patch_predictions[mask]).argmax()
            # Estimate probabilities from vote proportions
            votes = np.bincount(patch_predictions[mask], minlength=n_classes)
            sample_probabilities[i] = votes / votes.sum()
            
        elif aggregation_method == "max_prob":
            # Use patch with maximum probability
            if patch_probabilities.ndim > 1:
                max_idx = np.argmax(np.max(patch_probabilities[mask], axis=1))
                sample_probabilities[i] = patch_probabilities[mask][max_idx]
            else:
                max_idx = np.argmax(patch_probabilities[mask])
                sample_probabilities[i, 1] = patch_probabilities[mask][max_idx]
                sample_probabilities[i, 0] = 1 - sample_probabilities[i, 1]
            sample_predictions[i] = np.argmax(sample_probabilities[i])
    
    return sample_predictions, sample_probabilities, unique_samples
