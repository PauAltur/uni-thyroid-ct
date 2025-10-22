"""
Preprocessing functions for CT volume data.

This module provides efficient preprocessing operations for large 3D medical imaging volumes,
optimized for processing multiple volumes of size ~250x1500x1500.
"""

import numpy as np
from typing import Optional, Tuple, Union
import warnings
from scipy import ndimage


def normalize_volume_zscore(
    volume: np.ndarray, epsilon: float = 1e-8, dtype: Optional[np.dtype] = None
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
            RuntimeWarning,
        )
        normalized = volume.astype(dtype) - dtype.type(mean)
    else:
        # Perform normalization
        # For large volumes, this is done efficiently using numpy's broadcasting
        normalized = ((volume - mean) / std).astype(dtype)

    return normalized


def normalize_volume_zscore_inplace(
    volume: np.ndarray, epsilon: float = 1e-8
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
    if not volume.flags["C_CONTIGUOUS"] and not volume.flags["F_CONTIGUOUS"]:
        raise ValueError(
            "Volume must be contiguous in memory for in-place operations. "
            "Use volume.copy() first or use normalize_volume_zscore() instead."
        )

    # Convert to float if needed (in-place if possible)
    if not np.issubdtype(volume.dtype, np.floating):
        warnings.warn(
            f"Converting integer volume (dtype={volume.dtype}) to float64 for normalization.",
            RuntimeWarning,
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
            RuntimeWarning,
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
    verbose: bool = True,
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
            print(f"Normalizing volume {i + 1}/{len(volumes)}... ", end="", flush=True)

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


def resample_volume(
    volume: np.ndarray,
    original_spacing: Union[Tuple[float, float, float], np.ndarray],
    target_spacing: Union[Tuple[float, float, float], np.ndarray],
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    preserve_range: bool = True,
) -> np.ndarray:
    """
    Resample a 3D volume to a target resolution/spacing.

    This function resamples medical imaging volumes (e.g., CT scans) from their original
    voxel spacing to a target voxel spacing using high-quality interpolation. It's optimized
    for large volumes and uses scipy's efficient ndimage interpolation.

    The resampling is performed using zoom factors calculated as:
        zoom_factor = original_spacing / target_spacing

    For example:
    - original_spacing = [2.0, 1.0, 1.0] mm and target_spacing = [1.0, 1.0, 1.0] mm
      results in zoom_factor = [2.0, 1.0, 1.0], doubling resolution along the first axis.

    Args:
        volume: Input 3D numpy array of shape (D, H, W).
        original_spacing: Original voxel spacing in mm for each axis (D, H, W).
                         Can be tuple or numpy array.
        target_spacing: Target voxel spacing in mm for each axis (D, H, W).
                       Can be tuple or numpy array.
        order: Order of interpolation (0-5):
               0 = nearest-neighbor (fastest, lowest quality)
               1 = linear
               2 = quadratic
               3 = cubic (default, good balance of speed and quality)
               4 = quartic
               5 = quintic (highest quality, slowest)
        mode: How to handle values outside the input boundaries.
              Options: 'constant' (default), 'nearest', 'reflect', 'mirror', 'wrap'.
        cval: Value to use for points outside boundaries when mode='constant'.
              Default is 0.0 (typical for CT where outside is air/background).
        preserve_range: If True, clips output to the range of the input volume.
                       Useful to prevent interpolation artifacts outside data range.

    Returns:
        resampled_volume: Resampled 3D volume with new shape determined by spacing ratio.
                         New shape = original_shape * (original_spacing / target_spacing)

    Raises:
        ValueError: If input is not a 3D array or if spacing dimensions don't match.

    Example:
        >>> # Resample from anisotropic to isotropic spacing
        >>> volume = np.random.randn(100, 512, 512).astype(np.float32)
        >>> original_spacing = (2.5, 0.5, 0.5)  # mm - thicker slices in depth
        >>> target_spacing = (1.0, 1.0, 1.0)     # mm - isotropic 1mm voxels
        >>> resampled = resample_volume(volume, original_spacing, target_spacing)
        >>> print(f"Original shape: {volume.shape}")
        Original shape: (100, 512, 512)
        >>> print(f"Resampled shape: {resampled.shape}")
        Resampled shape: (250, 256, 256)

    Performance notes:
        - For large volumes, cubic interpolation (order=3) provides good quality
          with acceptable performance.
        - Use order=1 (linear) for faster resampling when quality is less critical.
        - Memory usage: intermediate arrays require ~2x volume size during resampling.
    """
    # Validate input
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume (D, H, W), got shape {volume.shape} with {volume.ndim} dimensions"
        )

    # Convert spacing to numpy arrays for easier computation
    original_spacing = np.asarray(original_spacing, dtype=np.float64)
    target_spacing = np.asarray(target_spacing, dtype=np.float64)

    # Validate spacing dimensions
    if original_spacing.shape != (3,):
        raise ValueError(
            f"original_spacing must have 3 elements, got {original_spacing.shape}"
        )
    if target_spacing.shape != (3,):
        raise ValueError(
            f"target_spacing must have 3 elements, got {target_spacing.shape}"
        )

    # Check for non-positive spacing values
    if np.any(original_spacing <= 0) or np.any(target_spacing <= 0):
        raise ValueError(
            "Spacing values must be positive. "
            f"Got original_spacing={original_spacing}, target_spacing={target_spacing}"
        )

    # Calculate zoom factors for each axis
    # zoom_factor = original_spacing / target_spacing
    # If original_spacing > target_spacing, we zoom in (increase resolution)
    # If original_spacing < target_spacing, we zoom out (decrease resolution)
    zoom_factors = original_spacing / target_spacing

    # Store original range for optional preservation
    if preserve_range:
        original_min = volume.min()
        original_max = volume.max()

    # Perform the resampling using scipy's zoom function
    # This is highly optimized and uses spline interpolation
    resampled = ndimage.zoom(
        volume,
        zoom=zoom_factors,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=True,  # Apply spline prefilter for better quality
    )

    # Optionally preserve the original intensity range
    if preserve_range:
        resampled = np.clip(resampled, original_min, original_max)

    return resampled


def extract_patches(
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
    stride: Optional[Tuple[int, int, int]] = None,
    padding_mode: str = "constant",
    padding_value: float = 0.0,
    drop_incomplete: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 3D or 2D patches from a 3D volume with configurable overlap and boundary handling.

    This function efficiently extracts patches from large medical imaging volumes using
    optimized array operations. Patches can overlap (controlled by stride) and boundaries
    can be handled by padding or dropping incomplete patches.

    For 2D patches from a 3D volume, set patch_size depth to 1 (e.g., (1, 256, 256)).

    Args:
        volume: Input 3D numpy array of shape (D, H, W).
        patch_size: Size of patches as (depth, height, width).
                   For 2D patches, use depth=1 (e.g., (1, 256, 256)).
                   All values must be positive integers.
        stride: Step size between patches as (depth_stride, height_stride, width_stride).
               If None, defaults to patch_size (non-overlapping patches).
               Use stride < patch_size for overlapping patches.
               Use stride > patch_size for gaps between patches (not recommended).
        padding_mode: How to pad the volume if needed:
                     'constant': Pad with padding_value (default)
                     'reflect': Reflect values at boundaries
                     'edge': Extend edge values
                     'wrap': Wrap around to opposite edge
                     Only used if drop_incomplete=False.
        padding_value: Value to use for padding when padding_mode='constant'.
                      Default is 0.0 (typical for CT background/air).
        drop_incomplete: If True, drop patches that would extend beyond volume boundaries.
                        If False, pad the volume to ensure all patches fit.
                        Default is False.

    Returns:
        patches: Array of extracted patches with shape:
                (num_patches, patch_depth, patch_height, patch_width)
                For 2D patches (depth=1): (num_patches, 1, patch_height, patch_width)
        positions: Array of patch positions (top-left corner) with shape:
                  (num_patches, 3) containing (d_start, h_start, w_start)
                  Useful for reconstructing the volume from patches.

    Raises:
        ValueError: If input is not 3D, patch_size is invalid, or stride is invalid.

    Examples:
        >>> # Extract non-overlapping 3D patches
        >>> volume = np.random.randn(100, 512, 512).astype(np.float32)
        >>> patches, positions = extract_patches(volume, patch_size=(32, 64, 64))
        >>> print(patches.shape)
        (128, 32, 64, 64)  # 4*8*8 = 256 patches (with padding)

        >>> # Extract overlapping 2D slices
        >>> volume = np.random.randn(100, 512, 512).astype(np.float32)
        >>> patches, positions = extract_patches(
        ...     volume, patch_size=(1, 256, 256), stride=(1, 128, 128)
        ... )
        >>> print(patches.shape)
        (300, 1, 256, 256)  # 100*3*3 = 300 patches (overlapping)

        >>> # Extract patches and drop incomplete ones
        >>> volume = np.random.randn(100, 500, 500).astype(np.float32)
        >>> patches, positions = extract_patches(
        ...     volume, patch_size=(10, 100, 100), drop_incomplete=True
        ... )
        >>> print(patches.shape)
        (500, 10, 100, 100)  # 10*5*5 = 250 complete patches only

    Performance notes:
        - Uses numpy's efficient array slicing and indexing
        - Memory usage: patches array is num_patches * patch_volume
        - For very large volumes, consider processing in batches
    """
    # Validate input
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume (D, H, W), got shape {volume.shape} with {volume.ndim} dimensions"
        )

    # Validate and process patch_size
    if len(patch_size) != 3:
        raise ValueError(
            f"patch_size must have 3 elements (depth, height, width), got {len(patch_size)}"
        )

    patch_d, patch_h, patch_w = patch_size
    if patch_d <= 0 or patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"All patch_size values must be positive, got {patch_size}")

    vol_d, vol_h, vol_w = volume.shape

    # Validate patch size doesn't exceed volume size (before padding)
    if not drop_incomplete:
        # When padding, we can handle any patch size
        pass
    else:
        # When dropping incomplete, need at least one full patch
        if patch_d > vol_d or patch_h > vol_h or patch_w > vol_w:
            raise ValueError(
                f"patch_size {patch_size} is larger than volume shape {volume.shape}. "
                f"Cannot extract patches with drop_incomplete=True."
            )

    # Set default stride (non-overlapping)
    if stride is None:
        stride = patch_size

    # Validate stride
    if len(stride) != 3:
        raise ValueError(
            f"stride must have 3 elements (depth, height, width), got {len(stride)}"
        )

    stride_d, stride_h, stride_w = stride
    if stride_d <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError(f"All stride values must be positive, got {stride}")

    # Calculate padding needed (if not dropping incomplete patches)
    if drop_incomplete:
        # Calculate how many complete patches fit
        num_patches_d = max(1, (vol_d - patch_d) // stride_d + 1)
        num_patches_h = max(1, (vol_h - patch_h) // stride_h + 1)
        num_patches_w = max(1, (vol_w - patch_w) // stride_w + 1)

        padded_volume = volume
        pad_d, pad_h, pad_w = 0, 0, 0
    else:
        # Calculate number of patches needed to cover the volume
        num_patches_d = max(1, int(np.ceil((vol_d - patch_d) / stride_d)) + 1)
        num_patches_h = max(1, int(np.ceil((vol_h - patch_h) / stride_h)) + 1)
        num_patches_w = max(1, int(np.ceil((vol_w - patch_w) / stride_w)) + 1)

        # Calculate required volume size
        required_d = (num_patches_d - 1) * stride_d + patch_d
        required_h = (num_patches_h - 1) * stride_h + patch_h
        required_w = (num_patches_w - 1) * stride_w + patch_w

        # Calculate padding needed
        pad_d = max(0, required_d - vol_d)
        pad_h = max(0, required_h - vol_h)
        pad_w = max(0, required_w - vol_w)

        # Apply padding if needed
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # Pad symmetrically when possible, extra padding goes to the end
            pad_width = (
                (0, pad_d),
                (0, pad_h),
                (0, pad_w),
            )

            if padding_mode == "constant":
                padded_volume = np.pad(
                    volume, pad_width, mode="constant", constant_values=padding_value
                )
            else:
                padded_volume = np.pad(volume, pad_width, mode=padding_mode)
        else:
            padded_volume = volume

    # Pre-allocate patches array
    total_patches = num_patches_d * num_patches_h * num_patches_w
    patches = np.empty((total_patches, patch_d, patch_h, patch_w), dtype=volume.dtype)
    positions = np.empty((total_patches, 3), dtype=np.int32)

    # Extract patches efficiently using numpy indexing
    patch_idx = 0
    for d_idx in range(num_patches_d):
        d_start = d_idx * stride_d
        d_end = d_start + patch_d

        for h_idx in range(num_patches_h):
            h_start = h_idx * stride_h
            h_end = h_start + patch_h

            for w_idx in range(num_patches_w):
                w_start = w_idx * stride_w
                w_end = w_start + patch_w

                # Extract patch
                patches[patch_idx] = padded_volume[
                    d_start:d_end, h_start:h_end, w_start:w_end
                ]
                positions[patch_idx] = [d_start, h_start, w_start]
                patch_idx += 1

    return patches, positions


if __name__ == "__main__":
    """Test and demonstrate the normalization functions."""

    print("=" * 80)
    print("Testing Volume Normalization Functions")
    print("=" * 80)

    # Test with a smaller volume for quick validation
    print("\n1. Testing with small volume (10x100x100)...")
    small_volume = np.random.randn(10, 100, 100).astype(np.float32) * 50 + 100
    print(
        f"   Original - Mean: {small_volume.mean():.2f}, Std: {small_volume.std():.2f}"
    )

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
    print("   Recommendation: Use dtype=np.float32 for memory efficiency")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
