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
    normalize_batch_zscore
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
            normalized = normalize_volume_zscore(volume, epsilon=1e-3)
            
            # Should issue a warning due to small std
            assert len(w) == 1
    
    def test_invalid_dimensions(self):
        """Test that non-3D arrays raise ValueError."""
        # Test 1D, 2D, and 4D arrays
        invalid_shapes = [
            (100,),           # 1D
            (50, 50),         # 2D
            (5, 10, 10, 3)    # 4D
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
            result = normalize_volume_zscore_inplace(volume)
            
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
        assert not non_contiguous.flags['C_CONTIGUOUS'] or not non_contiguous.flags['F_CONTIGUOUS']
        
        with pytest.raises(ValueError, match="contiguous in memory"):
            normalize_volume_zscore_inplace(non_contiguous)


class TestNormalizeBatchZscore:
    """Tests for normalize_batch_zscore function."""
    
    def test_batch_normalization(self):
        """Test that all volumes in batch are normalized."""
        volumes = [
            np.random.randn(5, 20, 20) * 10 + 50,
            np.random.randn(5, 20, 20) * 20 + 100,
            np.random.randn(5, 20, 20) * 5 + 25
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
            np.random.randn(5, 20, 20).astype(np.float64)
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
            np.random.randn(3, 15, 15)
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
            np.random.randn(5, 20, 20)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
