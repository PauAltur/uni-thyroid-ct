"""
Data loading utilities for classification tasks.

This module provides functions to load embeddings and metadata from HDF5 files,
filter patches by tissue percentage, and prepare datasets for classification.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


def load_embeddings_from_h5(
    h5_path: Path,
    embeddings_key: str = "embeddings",
    metadata_keys: Optional[List[str]] = None,
    min_tissue_percentage: float = 0.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load embeddings and metadata from an HDF5 file with optional tissue filtering.
    
    Args:
        h5_path: Path to the HDF5 file containing embeddings and metadata
        embeddings_key: Key for the embeddings dataset in the H5 file
        metadata_keys: List of metadata fields to load. If None, loads all available
        min_tissue_percentage: Minimum tissue percentage to include a patch (0-100)
        
    Returns:
        embeddings: Array of embeddings with shape (N, embedding_dim)
        metadata: DataFrame with metadata for each embedding
        
    Raises:
        FileNotFoundError: If H5 file doesn't exist
        KeyError: If required keys are not found in the H5 file
        ValueError: If min_tissue_percentage is out of valid range
    """
    # Validate tissue percentage range
    if not 0.0 <= min_tissue_percentage <= 100.0:
        raise ValueError(
            f"min_tissue_percentage must be in range [0, 100], got {min_tissue_percentage}"
        )
    
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    logger.info(f"Loading embeddings from {h5_path}")
    
    with h5py.File(h5_path, "r") as f:
        # Validate embeddings key exists
        if embeddings_key not in f:
            available_keys = list(f.keys())
            raise KeyError(
                f"Embeddings key '{embeddings_key}' not found. "
                f"Available keys: {available_keys}"
            )
        
        # Load embeddings
        embeddings = f[embeddings_key][()]
        logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
        
        # Determine metadata keys to load
        if metadata_keys is None:
            # Load all metadata fields
            metadata_keys = []
            if "metadata" in f:
                metadata_keys = list(f["metadata"].keys())
            # Also check for top-level metadata (e.g., sample_codes)
            for key in f.keys():
                if key not in [embeddings_key, "metadata"]:
                    metadata_keys.append(key)
        
        # Load metadata into dictionary
        metadata_dict = {}
        num_samples = embeddings.shape[0]
        
        for key in metadata_keys:
            # Try metadata group first, then top-level
            if "metadata" in f and key in f["metadata"]:
                data = f[f"metadata/{key}"][()]
            elif key in f:
                data = f[key][()]
            else:
                logger.warning(f"Metadata key '{key}' not found, skipping")
                continue
            
            # Handle different data types
            if data.dtype.type is np.bytes_:
                data = data.astype(str)
            
            # Ensure correct length
            if len(data) != num_samples:
                logger.warning(
                    f"Metadata '{key}' has length {len(data)}, "
                    f"expected {num_samples}. Skipping."
                )
                continue
            
            metadata_dict[key] = data
        
        # Create DataFrame
        metadata = pd.DataFrame(metadata_dict)
        logger.info(f"Loaded metadata with {len(metadata.columns)} fields")
    
    # Filter by tissue percentage if specified
    if min_tissue_percentage > 0.0:
        if "tissue_percentage" not in metadata.columns:
            logger.warning(
                f"Cannot filter by tissue percentage: 'tissue_percentage' "
                f"not found in metadata. Available: {metadata.columns.tolist()}"
            )
        else:
            initial_count = len(embeddings)
            mask = metadata["tissue_percentage"] >= min_tissue_percentage
            
            embeddings = embeddings[mask]
            metadata = metadata[mask].reset_index(drop=True)
            
            filtered_count = len(embeddings)
            logger.info(
                f"Filtered by tissue >= {min_tissue_percentage}%: "
                f"{initial_count} -> {filtered_count} patches "
                f"({filtered_count/initial_count*100:.1f}% retained)"
            )
    
    return embeddings, metadata


def prepare_classification_data(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    label_column: str,
    sample_column: str = "sample_code",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare embeddings and labels for classification with sample grouping.
    
    Args:
        embeddings: Array of embeddings (N, embedding_dim)
        metadata: DataFrame with labels and sample information
        label_column: Column name containing the target labels
        sample_column: Column name containing sample identifiers for grouping
        
    Returns:
        X: Feature matrix (embeddings)
        y: Target labels
        groups: Sample identifiers for group-aware splitting
        
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    # Validate columns exist
    if label_column not in metadata.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. "
            f"Available: {metadata.columns.tolist()}"
        )
    
    if sample_column not in metadata.columns:
        raise ValueError(
            f"Sample column '{sample_column}' not found. "
            f"Available: {metadata.columns.tolist()}"
        )
    
    # Validate sample column contains useful grouping information
    sample_values = metadata[sample_column].values
    unique_samples = np.unique(sample_values)
    
    if len(unique_samples) == len(sample_values):
        logger.warning(
            f"Sample column '{sample_column}' has all unique values! "
            f"Each patch has a unique identifier. GroupKFold will not prevent data leakage. "
            f"Consider using a different grouping column if patches from the same sample exist."
        )
    
    if len(unique_samples) == 1:
        raise ValueError(
            f"Sample column '{sample_column}' has only one unique value: {unique_samples[0]}. "
            f"Cannot perform group-based cross-validation."
        )
    
    # Check for missing values in critical columns
    if pd.isna(sample_values).any():
        n_missing = pd.isna(sample_values).sum()
        logger.warning(
            f"Sample column '{sample_column}' contains {n_missing} missing values. "
            f"These will be treated as a separate group."
        )
    
    # Validate shapes match
    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) "
            f"have different lengths"
        )
    
    # Extract labels
    labels = metadata[label_column].values
    
    # Handle boolean labels
    if labels.dtype == bool:
        labels = labels.astype(int)
    # Handle string labels
    elif labels.dtype.type is np.str_ or labels.dtype == object:
        # Convert to categorical codes
        unique_labels = np.unique(labels)
        logger.info(f"Converting string labels to integers. Classes: {unique_labels}")
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels])
    
    # Extract sample groups
    groups = metadata[sample_column].values
    
    # Log statistics
    unique_labels = np.unique(labels)
    unique_samples = np.unique(groups)
    
    logger.info("Dataset prepared:")
    logger.info(f"  Total patches: {len(embeddings)}")
    logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  Unique labels: {unique_labels}")
    logger.info(f"  Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    logger.info(f"  Unique samples: {len(unique_samples)}")
    logger.info(f"  Patches per sample (mean ± std): {len(labels)/len(unique_samples):.1f} ± {pd.Series(groups).value_counts().std():.1f}")
    
    return embeddings, labels, groups


def get_available_metadata_keys(h5_path: Path) -> Dict[str, List[str]]:
    """
    Get a list of available metadata keys in an HDF5 file.
    
    Args:
        h5_path: Path to the HDF5 file
        
    Returns:
        Dictionary with 'embeddings', 'metadata', and 'other' keys listing available fields
    """
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    
    available = {
        "embeddings": [],
        "metadata": [],
        "other": []
    }
    
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            if "embed" in key.lower():
                available["embeddings"].append(key)
            elif key == "metadata":
                # List all metadata subfields
                available["metadata"] = list(f["metadata"].keys())
            else:
                available["other"].append(key)
    
    return available
