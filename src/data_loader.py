"""
Data loading and filtering utilities for embeddings and metadata.

This module provides functions to load embeddings and metadata from HDF5 files
and apply filtering based on tissue percentage thresholds.
"""

import logging
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_embeddings_from_h5(
    h5_path: Path,
    tissue_threshold: float = 0.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Load embeddings and metadata from HDF5 file with optional tissue filtering.
    
    Args:
        h5_path: Path to HDF5 file containing embeddings and metadata
        tissue_threshold: Minimum tissue percentage to include patch (0-100)
    
    Returns:
        Tuple of (embeddings array, metadata DataFrame)
    
    Raises:
        FileNotFoundError: If HDF5 file does not exist
        KeyError: If required datasets are missing from HDF5 file
    
    Example:
        >>> embeddings, metadata = load_embeddings_from_h5(
        ...     Path("embeddings.h5"),
        ...     tissue_threshold=50.0
        ... )
        >>> print(f"Loaded {len(embeddings)} embeddings")
    """
    h5_path = Path(h5_path)
    
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    
    logger.info(f"Loading data from {h5_path}")
    
    with h5py.File(h5_path, "r") as f:
        # Verify required datasets exist
        required_datasets = [
            "embeddings",
            "sample_codes",
            "metadata/tissue_percentage",
            "metadata/has_invasion",
        ]
        
        missing = [ds for ds in required_datasets if ds not in f]
        if missing:
            raise KeyError(
                f"Missing required datasets in HDF5 file: {missing}\n"
                f"Available datasets: {list(f.keys())}"
            )
        
        # Load embeddings
        embeddings = f["embeddings"][:]
        logger.info(
            f"Loaded {len(embeddings):,} embeddings with "
            f"dimension {embeddings.shape[1]}"
        )
        
        # Load metadata
        sample_codes = f["sample_codes"][:].astype(str)
        tissue_pct = f["metadata/tissue_percentage"][:]
        has_invasion = f["metadata/has_invasion"][:]
        
        # Load optional invasion percentage if available
        if "metadata/invasion_percentage" in f:
            invasion_pct = f["metadata/invasion_percentage"][:]
        else:
            invasion_pct = np.zeros_like(has_invasion, dtype=float)
            logger.warning("invasion_percentage not found in HDF5, using zeros")
        
        # Create metadata DataFrame
        metadata = pd.DataFrame({
            "sample_code": sample_codes,
            "tissue_pct": tissue_pct,
            "has_invasion": has_invasion.astype(int),
            "invasion_pct": invasion_pct,
        })
    
    logger.info(f"Unique samples: {metadata['sample_code'].nunique()}")
    logger.info(
        f"Invasion prevalence: {metadata['has_invasion'].mean()*100:.2f}%"
    )
    
    # Apply tissue filtering if threshold > 0
    if tissue_threshold > 0:
        embeddings, metadata = filter_by_tissue_percentage(
            embeddings, metadata, tissue_threshold
        )
    
    return embeddings, metadata


def filter_by_tissue_percentage(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    tissue_threshold: float,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Filter patches based on tissue percentage threshold.
    
    Args:
        embeddings: Array of embeddings (n_patches, embedding_dim)
        metadata: DataFrame with metadata including 'tissue_pct' column
        tissue_threshold: Minimum tissue percentage to include (0-100)
    
    Returns:
        Tuple of (filtered_embeddings, filtered_metadata)
    
    Example:
        >>> embeddings_filtered, metadata_filtered = filter_by_tissue_percentage(
        ...     embeddings, metadata, tissue_threshold=30.0
        ... )
    """
    if tissue_threshold <= 0:
        logger.info("No tissue filtering applied (threshold <= 0)")
        return embeddings, metadata
    
    original_count = len(metadata)
    mask = metadata["tissue_pct"] >= tissue_threshold
    
    embeddings_filtered = embeddings[mask]
    metadata_filtered = metadata[mask].reset_index(drop=True)
    
    filtered_count = len(metadata_filtered)
    logger.info(
        f"Filtered patches by tissue >= {tissue_threshold}%: "
        f"{original_count:,} -> {filtered_count:,} "
        f"({filtered_count/original_count*100:.1f}% retained)"
    )
    logger.info(f"Remaining samples: {metadata_filtered['sample_code'].nunique()}")
    logger.info(
        f"Invasion prevalence after filtering: "
        f"{metadata_filtered['has_invasion'].mean()*100:.2f}%"
    )
    
    return embeddings_filtered, metadata_filtered


def get_sample_groups(metadata: pd.DataFrame) -> np.ndarray:
    """
    Extract sample group identifiers from metadata.
    
    Args:
        metadata: DataFrame with 'sample_code' column
    
    Returns:
        Array of sample codes for grouping
    
    Example:
        >>> groups = get_sample_groups(metadata)
        >>> print(f"Unique samples: {len(np.unique(groups))}")
    """
    return metadata["sample_code"].values


def get_labels(metadata: pd.DataFrame) -> np.ndarray:
    """
    Extract binary labels from metadata.
    
    Args:
        metadata: DataFrame with 'has_invasion' column
    
    Returns:
        Array of binary labels (0 or 1)
    
    Example:
        >>> labels = get_labels(metadata)
        >>> print(f"Positive class: {labels.sum()} / {len(labels)}")
    """
    return metadata["has_invasion"].values.astype(int)


def validate_data_for_cv(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    n_folds: int,
) -> None:
    """
    Validate that data is suitable for K-fold cross-validation.
    
    Args:
        embeddings: Array of embeddings
        metadata: DataFrame with metadata
        n_folds: Number of folds for cross-validation
    
    Raises:
        ValueError: If data is insufficient for requested number of folds
    
    Example:
        >>> validate_data_for_cv(embeddings, metadata, n_folds=5)
    """
    n_unique_samples = metadata["sample_code"].nunique()
    
    if n_unique_samples < n_folds:
        raise ValueError(
            f"Not enough unique samples ({n_unique_samples}) for {n_folds} folds. "
            f"Reduce n_folds or lower tissue_threshold."
        )
    
    logger.info(f"Validation passed: {n_unique_samples} samples for {n_folds} folds")
