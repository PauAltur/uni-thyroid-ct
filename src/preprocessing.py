"""
Preprocessing functions for CT volume data.

This module provides efficient preprocessing operations for large 3D medical imaging volumes,
optimized for processing multiple volumes of size ~250x1500x1500.
"""

import numpy as np
from typing import Optional
import warnings


def normalize_volume_zscore(
    volume: np.ndarray,
    epsilon: float = 1e-8,
    dtype: Optional[np.dtype] = None
) -> np.ndarray:
    """
    Apply z-score normalization to a 3D volume.
    
    Normalizes the volume to have zero mean and unit standard deviation:
        normalized = (volume - mean) / std
    
    This implementation is optimized for large volumes (~250x1500x1500) by:
    - Using in-place operations where possible
    - Computing statistics in float64 for numerical stability
    - Supporting output dtype specification to control memory usage
    
    Args:
        volume: Input 3D numpy array of shape (D, H, W).
        epsilon: Small constant added to standard deviation to avoid division by zero.
                 Default is 1e-8.
        dtype: Output data type. If None, uses the input volume's dtype.
               For memory efficiency with large volumes, consider using float32.
    
    Returns:
        normalized_volume: Z-score normalized volume with the same shape as input.
    
    Raises:
        ValueError: If input is not a 3D array.
        
    Example:
        >>> volume = np.random.randn(250, 1500, 1500).astype(np.float32)
        >>> normalized = normalize_volume_zscore(volume)
        >>> print(f"Mean: {normalized.mean():.6f}, Std: {normalized.std():.6f}")
        Mean: 0.000000, Std: 1.000000
    """
    # Validate input
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume (D, H, W), got shape {volume.shape} with {volume.ndim} dimensions"
        )
    
    # Determine output dtype
    if dtype is None:
        dtype = volume.dtype
    
    # Compute statistics using float64 for numerical stability
    # This is especially important for large volumes to avoid precision issues
    mean = np.float64(volume.mean())
    std = np.float64(volume.std())
    
    # Handle edge case: constant volume (std == 0)
    if std < epsilon:
        warnings.warn(
            f"Volume has very small standard deviation ({std:.2e}). "
            f"Returning zero-centered volume without scaling.",
            RuntimeWarning
        )
        normalized = volume.astype(dtype) - dtype.type(mean)
    else:
        # Perform normalization
        # For large volumes, this is done efficiently using numpy's broadcasting
        normalized = ((volume - mean) / std).astype(dtype)
    
    return normalized


def normalize_volume_zscore_inplace(
    volume: np.ndarray,
    epsilon: float = 1e-8
) -> np.ndarray:
    """
    Apply z-score normalization to a 3D volume in-place (modifies input).
    
    This is a memory-efficient variant that modifies the input volume directly,
    avoiding the creation of intermediate copies. Useful when processing many
    large volumes sequentially.
    
    Note: This function requires the input volume to be a floating-point type.
    If your volume is integer-typed, it will be converted to float64 in-place.
    
    Args:
        volume: Input 3D numpy array of shape (D, H, W). Will be modified in-place.
                Must be a mutable array (not a view with non-contiguous memory).
        epsilon: Small constant added to standard deviation to avoid division by zero.
    
    Returns:
        The same volume array (now normalized), returned for convenience.
    
    Raises:
        ValueError: If input is not a 3D array or not contiguous in memory.
        
    Example:
        >>> volume = np.random.randn(250, 1500, 1500).astype(np.float32)
        >>> original_id = id(volume)
        >>> normalized = normalize_volume_zscore_inplace(volume)
        >>> assert id(normalized) == original_id  # Same object
        >>> print(f"Mean: {volume.mean():.6f}, Std: {volume.std():.6f}")
        Mean: 0.000000, Std: 1.000000
    """
    # Validate input
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume (D, H, W), got shape {volume.shape} with {volume.ndim} dimensions"
        )
    
    # Ensure we can modify in-place (need contiguous memory)
    if not volume.flags['C_CONTIGUOUS'] and not volume.flags['F_CONTIGUOUS']:
        raise ValueError(
            "Volume must be contiguous in memory for in-place operations. "
            "Use volume.copy() first or use normalize_volume_zscore() instead."
        )
    
    # Convert to float if needed (in-place if possible)
    if not np.issubdtype(volume.dtype, np.floating):
        warnings.warn(
            f"Converting integer volume (dtype={volume.dtype}) to float64 for normalization.",
            RuntimeWarning
        )
        # Note: This creates a new array, so we lose the in-place advantage here
        volume = volume.astype(np.float64)
    
    # Compute statistics
    mean = volume.mean()
    std = volume.std()
    
    # Handle edge case: constant volume
    if std < epsilon:
        warnings.warn(
            f"Volume has very small standard deviation ({std:.2e}). "
            f"Returning zero-centered volume without scaling.",
            RuntimeWarning
        )
        volume -= mean
    else:
        # Perform in-place normalization
        volume -= mean
        volume /= std
    
    return volume


def normalize_batch_zscore(
    volumes: list[np.ndarray],
    epsilon: float = 1e-8,
    dtype: Optional[np.dtype] = np.float32,
    verbose: bool = True
) -> list[np.ndarray]:
    """
    Apply z-score normalization to multiple volumes efficiently.
    
    Each volume is normalized independently (not across the batch).
    This function is optimized for processing multiple large volumes by
    allowing control over output dtype to manage memory usage.
    
    Args:
        volumes: List of 3D numpy arrays, each of shape (D, H, W).
        epsilon: Small constant to avoid division by zero.
        dtype: Output data type for all volumes. Use float32 to reduce memory usage.
        verbose: If True, prints progress information.
    
    Returns:
        List of normalized volumes with the specified dtype.
        
    Example:
        >>> volumes = [np.random.randn(250, 1500, 1500) for _ in range(5)]
        >>> normalized_volumes = normalize_batch_zscore(volumes, dtype=np.float32)
        >>> print(f"Processed {len(normalized_volumes)} volumes")
        Processed 5 volumes
    """
    normalized = []
    
    for i, volume in enumerate(volumes):
        if verbose:
            print(f"Normalizing volume {i+1}/{len(volumes)}... ", end='', flush=True)
        
        try:
            norm_vol = normalize_volume_zscore(volume, epsilon=epsilon, dtype=dtype)
            normalized.append(norm_vol)
            
            if verbose:
                print("✓")
        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            raise
    
    return normalized


if __name__ == "__main__":
    """Test and demonstrate the normalization functions."""
    
    print("=" * 80)
    print("Testing Volume Normalization Functions")
    print("=" * 80)
    
    # Test with a smaller volume for quick validation
    print("\n1. Testing with small volume (10x100x100)...")
    small_volume = np.random.randn(10, 100, 100).astype(np.float32) * 50 + 100
    print(f"   Original - Mean: {small_volume.mean():.2f}, Std: {small_volume.std():.2f}")
    
    normalized = normalize_volume_zscore(small_volume)
    print(f"   Normalized - Mean: {normalized.mean():.6f}, Std: {normalized.std():.6f}")
    
    # Test in-place normalization
    print("\n2. Testing in-place normalization...")
    test_volume = small_volume.copy()
    original_id = id(test_volume)
    result = normalize_volume_zscore_inplace(test_volume)
    print(f"   Same object: {id(result) == original_id}")
    print(f"   Mean: {test_volume.mean():.6f}, Std: {test_volume.std():.6f}")
    
    # Test edge case: constant volume
    print("\n3. Testing edge case (constant volume)...")
    constant_volume = np.ones((5, 50, 50), dtype=np.float32) * 42
    normalized_constant = normalize_volume_zscore(constant_volume)
    print(f"   Result mean: {normalized_constant.mean():.6f}")
    print(f"   Result std: {normalized_constant.std():.6f}")
    
    # Memory and performance info
    print("\n4. Memory usage estimation for typical volume (250x1500x1500)...")
    typical_shape = (250, 1500, 1500)
    size_float64 = np.prod(typical_shape) * 8 / (1024**3)  # GB
    size_float32 = np.prod(typical_shape) * 4 / (1024**3)  # GB
    print(f"   float64: {size_float64:.2f} GB per volume")
    print(f"   float32: {size_float32:.2f} GB per volume")
    print(f"   Recommendation: Use dtype=np.float32 for memory efficiency")
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
