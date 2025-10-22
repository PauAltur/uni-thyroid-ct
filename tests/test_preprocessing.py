"""
Tests for preprocessing functions.

Run with: pytest tests/test_preprocessing.py -v
"""

import pytest
import numpy as np
import warnings
from src.preprocessing import (
    normalize_volume_zscore,
    normalize_volume_zscore_inplace,
    normalize_batch_zscore,
    resample_volume,
)


class TestNormalizeVolumeZscore:
    """Tests for normalize_volume_zscore function."""

    def test_basic_normalization(self):
        """Test that normalization produces zero mean and unit std."""
        volume = np.random.randn(10, 50, 50).astype(np.float32) * 10 + 50
        normalized = normalize_volume_zscore(volume)

        assert np.isclose(normalized.mean(), 0.0, atol=1e-5)
        assert np.isclose(normalized.std(), 1.0, atol=1e-5)

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        shapes = [(5, 10, 10), (20, 100, 100), (50, 200, 200)]
        for shape in shapes:
            volume = np.random.randn(*shape)
            normalized = normalize_volume_zscore(volume)
            assert normalized.shape == shape

    def test_dtype_preservation(self):
        """Test that dtype is preserved when not specified."""
        for dtype in [np.float32, np.float64]:
            volume = np.random.randn(5, 20, 20).astype(dtype)
            normalized = normalize_volume_zscore(volume)
            assert normalized.dtype == dtype

    def test_dtype_conversion(self):
        """Test explicit dtype conversion."""
        volume = np.random.randn(5, 20, 20).astype(np.float64)
        normalized = normalize_volume_zscore(volume, dtype=np.float32)
        assert normalized.dtype == np.float32

    def test_constant_volume(self):
        """Test handling of constant volume (std=0)."""
        volume = np.ones((5, 20, 20), dtype=np.float32) * 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalized = normalize_volume_zscore(volume)

            # Should issue a warning
            assert len(w) == 1
            assert issubclass(w[-1].category, RuntimeWarning)
            assert "small standard deviation" in str(w[-1].message).lower()

        # Result should be zero-centered
        assert np.allclose(normalized, 0.0)

    def test_nearly_constant_volume(self):
        """Test handling of nearly constant volume."""
        volume = np.ones((5, 20, 20)) * 100
        volume[0, 0, 0] = 100.0001  # Tiny variation

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = normalize_volume_zscore(volume, epsilon=1e-3)

            # Should issue a warning due to small std
            assert len(w) == 1

    def test_invalid_dimensions(self):
        """Test that non-3D arrays raise ValueError."""
        # Test 1D, 2D, and 4D arrays
        invalid_shapes = [
            (100,),  # 1D
            (50, 50),  # 2D
            (5, 10, 10, 3),  # 4D
        ]

        for shape in invalid_shapes:
            volume = np.random.randn(*shape)
            with pytest.raises(ValueError, match="Expected 3D volume"):
                normalize_volume_zscore(volume)

    def test_numerical_stability(self):
        """Test normalization with very large and very small values."""
        # Very large values - use more moderate scaling to avoid precision issues
        large_volume = np.random.randn(5, 20, 20) * 1e6 + 1e8
        normalized_large = normalize_volume_zscore(large_volume)
        assert np.isclose(normalized_large.mean(), 0.0, atol=1e-2)
        assert np.isclose(normalized_large.std(), 1.0, atol=1e-2)

        # Very small values
        small_volume = np.random.randn(5, 20, 20) * 1e-6 + 1e-8
        normalized_small = normalize_volume_zscore(small_volume)
        assert np.isclose(normalized_small.mean(), 0.0, atol=1e-2)
        assert np.isclose(normalized_small.std(), 1.0, atol=1e-2)

    def test_with_negative_values(self):
        """Test normalization with negative values."""
        volume = np.random.randn(5, 20, 20) * 50 - 100  # Mean around -100
        normalized = normalize_volume_zscore(volume)

        assert np.isclose(normalized.mean(), 0.0, atol=1e-5)
        assert np.isclose(normalized.std(), 1.0, atol=1e-5)

    def test_epsilon_parameter(self):
        """Test that epsilon parameter is respected."""
        volume = np.ones((5, 20, 20)) * 10

        # With default epsilon (1e-8), should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_volume_zscore(volume)
            assert len(w) == 1

        # With larger epsilon, should also warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_volume_zscore(volume, epsilon=1e-5)
            assert len(w) == 1


class TestNormalizeVolumeZscoreInplace:
    """Tests for normalize_volume_zscore_inplace function."""

    def test_inplace_modification(self):
        """Test that the function modifies the input array."""
        volume = np.random.randn(5, 20, 20).astype(np.float32) * 10 + 50
        original_id = id(volume)
        original_mean = volume.mean()

        result = normalize_volume_zscore_inplace(volume)

        # Should be the same object
        assert id(result) == original_id
        assert id(volume) == original_id

        # Should be normalized
        assert np.isclose(volume.mean(), 0.0, atol=1e-5)
        assert np.isclose(volume.std(), 1.0, atol=1e-5)

        # Mean should have changed
        assert not np.isclose(volume.mean(), original_mean)

    def test_inplace_with_float32(self):
        """Test in-place normalization with float32."""
        volume = np.random.randn(5, 20, 20).astype(np.float32)
        normalize_volume_zscore_inplace(volume)

        assert volume.dtype == np.float32
        assert np.isclose(volume.mean(), 0.0, atol=1e-5)
        assert np.isclose(volume.std(), 1.0, atol=1e-5)

    def test_inplace_with_integer_warns(self):
        """Test that integer arrays trigger a warning."""
        volume = np.random.randint(0, 255, (5, 20, 20), dtype=np.uint8)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = normalize_volume_zscore_inplace(volume)

            # Should warn about conversion
            assert len(w) >= 1
            assert any("Converting integer" in str(warning.message) for warning in w)

    def test_inplace_invalid_dimensions(self):
        """Test that non-3D arrays raise ValueError."""
        volume = np.random.randn(50, 50).astype(np.float32)  # 2D

        with pytest.raises(ValueError, match="Expected 3D volume"):
            normalize_volume_zscore_inplace(volume)

    def test_inplace_constant_volume(self):
        """Test in-place normalization of constant volume."""
        volume = np.ones((5, 20, 20), dtype=np.float32) * 42

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalize_volume_zscore_inplace(volume)
            assert len(w) == 1  # Should warn about small std

        assert np.allclose(volume, 0.0)

    def test_inplace_contiguous_check(self):
        """Test that non-contiguous arrays raise ValueError."""
        volume = np.random.randn(10, 30, 30).astype(np.float32)
        # Create a non-contiguous view
        non_contiguous = volume[::2, ::2, ::2]

        # Check that it's actually non-contiguous
        assert (
            not non_contiguous.flags["C_CONTIGUOUS"]
            or not non_contiguous.flags["F_CONTIGUOUS"]
        )

        with pytest.raises(ValueError, match="contiguous in memory"):
            normalize_volume_zscore_inplace(non_contiguous)


class TestNormalizeBatchZscore:
    """Tests for normalize_batch_zscore function."""

    def test_batch_normalization(self):
        """Test that all volumes in batch are normalized."""
        volumes = [
            np.random.randn(5, 20, 20) * 10 + 50,
            np.random.randn(5, 20, 20) * 20 + 100,
            np.random.randn(5, 20, 20) * 5 + 25,
        ]

        normalized = normalize_batch_zscore(volumes, verbose=False)

        assert len(normalized) == len(volumes)

        for norm_vol in normalized:
            assert np.isclose(norm_vol.mean(), 0.0, atol=1e-5)
            assert np.isclose(norm_vol.std(), 1.0, atol=1e-5)

    def test_batch_dtype_conversion(self):
        """Test that all volumes are converted to specified dtype."""
        volumes = [
            np.random.randn(5, 20, 20).astype(np.float64),
            np.random.randn(5, 20, 20).astype(np.float64),
        ]

        normalized = normalize_batch_zscore(volumes, dtype=np.float32, verbose=False)

        for norm_vol in normalized:
            assert norm_vol.dtype == np.float32

    def test_batch_empty_list(self):
        """Test that empty list returns empty list."""
        normalized = normalize_batch_zscore([], verbose=False)
        assert normalized == []

    def test_batch_single_volume(self):
        """Test with a single volume in the batch."""
        volumes = [np.random.randn(5, 20, 20)]
        normalized = normalize_batch_zscore(volumes, verbose=False)

        assert len(normalized) == 1
        assert np.isclose(normalized[0].mean(), 0.0, atol=1e-5)

    def test_batch_different_shapes(self):
        """Test batch with different volume shapes."""
        volumes = [
            np.random.randn(5, 20, 20),
            np.random.randn(10, 30, 30),
            np.random.randn(3, 15, 15),
        ]

        normalized = normalize_batch_zscore(volumes, verbose=False)

        assert len(normalized) == len(volumes)

        for orig, norm in zip(volumes, normalized):
            assert norm.shape == orig.shape
            assert np.isclose(norm.mean(), 0.0, atol=1e-5)

    def test_batch_with_invalid_volume(self):
        """Test that batch fails if one volume is invalid."""
        volumes = [
            np.random.randn(5, 20, 20),
            np.random.randn(50, 50),  # 2D - invalid!
            np.random.randn(5, 20, 20),
        ]

        with pytest.raises(ValueError):
            normalize_batch_zscore(volumes, verbose=False)

    def test_batch_verbose_output(self, capsys):
        """Test that verbose mode produces output."""
        volumes = [np.random.randn(5, 20, 20) for _ in range(3)]

        normalize_batch_zscore(volumes, verbose=True)

        captured = capsys.readouterr()
        assert "Normalizing volume" in captured.out
        assert "✓" in captured.out


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_large_volume_shape(self):
        """Test with a large volume (memory permitting)."""
        # Create a moderately large volume for testing
        # Actual size 250x1500x1500 would be ~5GB, so use smaller
        volume = np.random.randn(50, 300, 300).astype(np.float32)
        normalized = normalize_volume_zscore(volume, dtype=np.float32)

        assert normalized.shape == volume.shape
        assert np.isclose(normalized.mean(), 0.0, atol=1e-4)
        assert np.isclose(normalized.std(), 1.0, atol=1e-4)

    def test_zero_volume(self):
        """Test with a volume of all zeros."""
        volume = np.zeros((5, 20, 20), dtype=np.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            normalized = normalize_volume_zscore(volume)
            assert len(w) == 1  # Should warn

        assert np.allclose(normalized, 0.0)

    def test_nan_handling(self):
        """Test that NaN values are handled appropriately."""
        volume = np.random.randn(5, 20, 20).astype(np.float32)
        volume[0, 0, 0] = np.nan

        normalized = normalize_volume_zscore(volume)

        # NaN should propagate
        assert np.isnan(normalized[0, 0, 0])

    def test_inf_handling(self):
        """Test that infinity values result in NaN in normalized output."""
        volume = np.random.randn(5, 20, 20).astype(np.float32)
        volume[0, 0, 0] = np.inf

        normalized = normalize_volume_zscore(volume)

        # When mean or std contains inf, normalization produces NaN
        # This is expected behavior as (inf - inf) = nan and (value / inf) = 0 or nan
        assert np.isnan(normalized).any() or np.isinf(normalized).any()


class TestResampleVolume:
    """Tests for resample_volume function."""

    def test_basic_resampling_isotropic_upsampling(self):
        """Test upsampling from anisotropic to isotropic spacing."""
        # Create a simple volume
        volume = np.random.randn(50, 100, 100).astype(np.float32)
        original_spacing = (2.0, 1.0, 1.0)  # 2mm slices, 1mm in-plane
        target_spacing = (1.0, 1.0, 1.0)  # Isotropic 1mm

        resampled = resample_volume(volume, original_spacing, target_spacing)

        # Check shape: should double in first dimension
        expected_shape = (100, 100, 100)
        assert resampled.shape == expected_shape

    def test_basic_resampling_downsampling(self):
        """Test downsampling to lower resolution."""
        volume = np.random.randn(100, 200, 200).astype(np.float32)
        original_spacing = (1.0, 1.0, 1.0)
        target_spacing = (2.0, 2.0, 2.0)  # Downsample by factor of 2

        resampled = resample_volume(volume, original_spacing, target_spacing)

        # Check shape: should halve in all dimensions
        expected_shape = (50, 100, 100)
        assert resampled.shape == expected_shape

    def test_output_dtype_preservation(self):
        """Test that output dtype matches input dtype."""
        for dtype in [np.float32, np.float64]:
            volume = np.random.randn(20, 50, 50).astype(dtype)
            original_spacing = (2.0, 1.0, 1.0)
            target_spacing = (1.0, 1.0, 1.0)

            resampled = resample_volume(volume, original_spacing, target_spacing)
            assert resampled.dtype == dtype

    def test_no_resampling_when_spacing_equal(self):
        """Test that volume is unchanged when spacings are equal."""
        volume = np.random.randn(30, 60, 60).astype(np.float32)
        spacing = (1.5, 1.0, 1.0)

        resampled = resample_volume(volume, spacing, spacing)

        # Shape should be identical
        assert resampled.shape == volume.shape
        # Values should be very close (minor interpolation artifacts may occur)
        assert np.allclose(resampled, volume, rtol=1e-5, atol=1e-6)

    def test_anisotropic_resampling(self):
        """Test resampling with different factors in each dimension."""
        volume = np.random.randn(40, 80, 120).astype(np.float32)
        original_spacing = (2.5, 1.0, 0.5)
        target_spacing = (1.0, 2.0, 1.0)

        resampled = resample_volume(volume, original_spacing, target_spacing)

        # Calculate expected shape
        # zoom = original / target = [2.5, 0.5, 0.5]
        # new_shape = [40*2.5, 80*0.5, 120*0.5] = [100, 40, 60]
        expected_shape = (100, 40, 60)
        assert resampled.shape == expected_shape

    def test_interpolation_orders(self):
        """Test different interpolation orders."""
        volume = np.random.randn(20, 40, 40).astype(np.float32)
        original_spacing = (2.0, 1.0, 1.0)
        target_spacing = (1.0, 1.0, 1.0)

        # Test all interpolation orders
        for order in range(6):  # 0 to 5
            resampled = resample_volume(
                volume, original_spacing, target_spacing, order=order
            )
            assert resampled.shape == (40, 40, 40)

    def test_nearest_neighbor_interpolation(self):
        """Test that nearest-neighbor preserves exact values."""
        # Create a volume with distinct integer values
        volume = np.arange(27).reshape(3, 3, 3).astype(np.float32)
        original_spacing = (2.0, 2.0, 2.0)
        target_spacing = (2.0, 2.0, 2.0)  # No change

        resampled = resample_volume(volume, original_spacing, target_spacing, order=0)

        # With order=0 and same spacing, values should be identical
        assert np.array_equal(resampled, volume)

    def test_preserve_range_true(self):
        """Test that preserve_range=True clips to original range."""
        volume = np.random.randn(20, 40, 40).astype(np.float32) * 100 + 500
        original_min = volume.min()
        original_max = volume.max()

        original_spacing = (2.0, 1.0, 1.0)
        target_spacing = (1.0, 0.5, 0.5)

        resampled = resample_volume(
            volume, original_spacing, target_spacing, preserve_range=True
        )

        # Values should be within original range
        assert resampled.min() >= original_min
        assert resampled.max() <= original_max

    def test_preserve_range_false(self):
        """Test that preserve_range=False may exceed original range."""
        volume = np.random.randn(20, 40, 40).astype(np.float32) * 100 + 500

        original_spacing = (2.0, 1.0, 1.0)
        target_spacing = (1.0, 0.5, 0.5)

        resampled = resample_volume(
            volume, original_spacing, target_spacing, preserve_range=False
        )

        # May have interpolation artifacts outside range (though not guaranteed)
        # At minimum, should not crash and should have reasonable values
        assert np.isfinite(resampled).all()

    def test_mode_constant(self):
        """Test constant mode with specific boundary value."""
        volume = np.ones((10, 20, 20), dtype=np.float32) * 100
        original_spacing = (1.0, 1.0, 1.0)
        target_spacing = (0.5, 0.5, 0.5)  # Upsample

        resampled = resample_volume(
            volume, original_spacing, target_spacing, mode="constant", cval=-1000
        )

        assert resampled.shape == (20, 40, 40)
        # Most values should still be around 100, edges might be affected
        assert np.median(resampled) > 50  # Reasonable check

    def test_mode_nearest(self):
        """Test nearest mode for boundary handling."""
        volume = np.random.randn(10, 20, 20).astype(np.float32)
        original_spacing = (1.0, 1.0, 1.0)
        target_spacing = (0.8, 0.8, 0.8)

        resampled = resample_volume(
            volume, original_spacing, target_spacing, mode="nearest"
        )

        # Should not have constant boundary values
        assert np.isfinite(resampled).all()

    def test_invalid_dimensions(self):
        """Test that non-3D arrays raise ValueError."""
        invalid_shapes = [
            (100,),  # 1D
            (50, 50),  # 2D
            (5, 10, 10, 3),  # 4D
        ]

        for shape in invalid_shapes:
            volume = np.random.randn(*shape)
            with pytest.raises(ValueError, match="Expected 3D volume"):
                resample_volume(volume, (1, 1, 1), (2, 2, 2))

    def test_invalid_spacing_dimensions(self):
        """Test that spacing with wrong dimensions raises ValueError."""
        volume = np.random.randn(10, 20, 20)

        # Test wrong number of spacing values
        with pytest.raises(ValueError, match="must have 3 elements"):
            resample_volume(volume, (1, 1), (1, 1, 1))

        with pytest.raises(ValueError, match="must have 3 elements"):
            resample_volume(volume, (1, 1, 1), (1, 1))

        with pytest.raises(ValueError, match="must have 3 elements"):
            resample_volume(volume, (1, 1, 1, 1), (1, 1, 1))

    def test_negative_spacing_values(self):
        """Test that negative spacing values raise ValueError."""
        volume = np.random.randn(10, 20, 20)

        with pytest.raises(ValueError, match="must be positive"):
            resample_volume(volume, (-1, 1, 1), (1, 1, 1))

        with pytest.raises(ValueError, match="must be positive"):
            resample_volume(volume, (1, 1, 1), (1, -1, 1))

    def test_zero_spacing_values(self):
        """Test that zero spacing values raise ValueError."""
        volume = np.random.randn(10, 20, 20)

        with pytest.raises(ValueError, match="must be positive"):
            resample_volume(volume, (0, 1, 1), (1, 1, 1))

        with pytest.raises(ValueError, match="must be positive"):
            resample_volume(volume, (1, 1, 1), (0, 1, 1))

    def test_spacing_as_numpy_array(self):
        """Test that spacing can be provided as numpy arrays."""
        volume = np.random.randn(20, 40, 40).astype(np.float32)
        original_spacing = np.array([2.0, 1.0, 1.0])
        target_spacing = np.array([1.0, 1.0, 1.0])

        resampled = resample_volume(volume, original_spacing, target_spacing)

        expected_shape = (40, 40, 40)
        assert resampled.shape == expected_shape

    def test_spacing_as_list(self):
        """Test that spacing can be provided as lists."""
        volume = np.random.randn(20, 40, 40).astype(np.float32)
        original_spacing = [2.0, 1.0, 1.0]
        target_spacing = [1.0, 1.0, 1.0]

        resampled = resample_volume(volume, original_spacing, target_spacing)

        expected_shape = (40, 40, 40)
        assert resampled.shape == expected_shape

    def test_large_upsampling_factor(self):
        """Test resampling with large upsampling factor."""
        volume = np.random.randn(5, 10, 10).astype(np.float32)
        original_spacing = (5.0, 5.0, 5.0)
        target_spacing = (1.0, 1.0, 1.0)

        resampled = resample_volume(volume, original_spacing, target_spacing)

        expected_shape = (25, 50, 50)
        assert resampled.shape == expected_shape

    def test_large_downsampling_factor(self):
        """Test resampling with large downsampling factor."""
        volume = np.random.randn(100, 100, 100).astype(np.float32)
        original_spacing = (1.0, 1.0, 1.0)
        target_spacing = (5.0, 5.0, 5.0)

        resampled = resample_volume(volume, original_spacing, target_spacing)

        expected_shape = (20, 20, 20)
        assert resampled.shape == expected_shape

    def test_realistic_ct_spacing(self):
        """Test with realistic CT scan spacing values."""
        # Typical CT: thick slices, thin in-plane resolution
        volume = np.random.randn(150, 512, 512).astype(np.float32)
        original_spacing = (2.5, 0.488, 0.488)  # Typical CT spacing
        target_spacing = (1.0, 1.0, 1.0)  # Isotropic 1mm

        resampled = resample_volume(volume, original_spacing, target_spacing)

        # Expected: [150 * 2.5, 512 * 0.488, 512 * 0.488]
        #         ≈ [375, 250, 250]
        assert resampled.shape[0] > volume.shape[0]  # Upsampled in depth
        assert resampled.shape[1] < volume.shape[1]  # Downsampled in-plane
        assert resampled.shape[2] < volume.shape[2]

    def test_constant_volume_resampling(self):
        """Test resampling a constant volume."""
        volume = np.ones((20, 30, 30), dtype=np.float32) * 42
        original_spacing = (2.0, 1.0, 1.0)
        target_spacing = (1.0, 1.0, 1.0)

        resampled = resample_volume(volume, original_spacing, target_spacing)

        # Constant volume should remain constant
        assert np.allclose(resampled, 42.0, rtol=1e-5)

    def test_interpolation_quality_linear_gradient(self):
        """Test interpolation quality on a linear gradient."""
        # Create a linear gradient in the first dimension
        volume = np.zeros((20, 30, 30), dtype=np.float32)
        for i in range(20):
            volume[i, :, :] = i * 10

        original_spacing = (2.0, 1.0, 1.0)
        target_spacing = (1.0, 1.0, 1.0)

        resampled = resample_volume(volume, original_spacing, target_spacing, order=1)

        # With linear interpolation, gradient should be preserved
        # Check that values increase monotonically along first axis
        for i in range(1, resampled.shape[0]):
            assert resampled[i, 15, 15] >= resampled[i - 1, 15, 15]

    def test_memory_efficiency_dtype(self):
        """Test that resampling maintains memory-efficient dtypes."""
        # Use float32 for memory efficiency
        volume = np.random.randn(30, 60, 60).astype(np.float32)
        original_spacing = (2.0, 1.0, 1.0)
        target_spacing = (1.0, 1.0, 1.0)

        resampled = resample_volume(volume, original_spacing, target_spacing)

        assert resampled.dtype == np.float32

        # Verify memory usage is reasonable
        volume_size_mb = volume.nbytes / (1024**2)
        resampled_size_mb = resampled.nbytes / (1024**2)
        # Resampled should be ~2x larger (doubled in first dimension)
        assert resampled_size_mb < volume_size_mb * 3  # Some margin for safety


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
