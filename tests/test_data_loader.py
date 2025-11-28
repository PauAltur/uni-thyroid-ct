"""
Tests for data_loader module.

Tests data loading, filtering, and validation functionality.
"""

import pytest
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    load_embeddings_from_h5,
    filter_by_tissue_percentage,
    get_sample_groups,
    get_labels,
    validate_data_for_cv,
)


@pytest.fixture
def sample_h5_file():
    """Create a temporary H5 file with sample data."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
    temp_file.close()  # Close immediately to avoid permission issues on Windows
    
    # Create sample data
    n_patches = 100
    embedding_dim = 1024
    
    embeddings = np.random.randn(n_patches, embedding_dim).astype(np.float32)
    sample_codes = np.array([f"SAMPLE_{i//10:03d}" for i in range(n_patches)])
    tissue_pct = np.random.uniform(0, 100, n_patches).astype(np.float32)
    has_invasion = np.random.randint(0, 2, n_patches).astype(np.int32)
    invasion_pct = np.where(has_invasion, np.random.uniform(0, 100, n_patches), 0).astype(np.float32)
    
    # Write to H5 file
    with h5py.File(temp_file.name, 'w') as f:
        f.create_dataset('embeddings', data=embeddings)
        f.create_group('metadata')
        f['metadata'].create_dataset('sample_code', data=sample_codes.astype('S'))
        f['metadata'].create_dataset('tissue_percentage', data=tissue_pct)
        f['metadata'].create_dataset('has_invasion', data=has_invasion)
        f['metadata'].create_dataset('invasion_percentage', data=invasion_pct)
    
    yield Path(temp_file.name)
    
    # Cleanup
    try:
        Path(temp_file.name).unlink()
    except PermissionError:
        pass  # Ignore cleanup errors on Windows


def test_load_embeddings_from_h5_basic(sample_h5_file):
    """Test basic loading of embeddings and metadata."""
    embeddings, metadata = load_embeddings_from_h5(sample_h5_file, tissue_threshold=0.0)
    
    assert isinstance(embeddings, np.ndarray)
    assert isinstance(metadata, pd.DataFrame)
    assert embeddings.shape[0] == 100
    assert embeddings.shape[1] == 1024
    assert len(metadata) == 100
    assert 'sample_code' in metadata.columns
    assert 'tissue_pct' in metadata.columns
    assert 'has_invasion' in metadata.columns
    assert 'invasion_pct' in metadata.columns


def test_load_embeddings_with_tissue_filtering(sample_h5_file):
    """Test loading with tissue percentage filtering."""
    embeddings, metadata = load_embeddings_from_h5(sample_h5_file, tissue_threshold=50.0)
    
    assert len(embeddings) == len(metadata)
    assert len(embeddings) < 100  # Should filter some patches
    assert all(metadata['tissue_pct'] >= 50.0)


def test_load_embeddings_file_not_found():
    """Test error handling when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_embeddings_from_h5(Path("nonexistent_file.h5"))


def test_filter_by_tissue_percentage():
    """Test tissue percentage filtering."""
    embeddings = np.random.randn(100, 10)
    metadata = pd.DataFrame({
        'sample_code': [f"S{i:03d}" for i in range(100)],
        'tissue_pct': np.random.uniform(0, 100, 100),
        'has_invasion': np.random.randint(0, 2, 100),
        'invasion_pct': np.random.uniform(0, 50, 100),
    })
    
    # Filter with threshold 50
    filtered_embeddings, filtered_metadata = filter_by_tissue_percentage(
        embeddings, metadata, tissue_threshold=50.0
    )
    
    assert len(filtered_embeddings) == len(filtered_metadata)
    assert len(filtered_embeddings) <= len(embeddings)
    assert all(filtered_metadata['tissue_pct'] >= 50.0)


def test_filter_by_tissue_zero_threshold():
    """Test that zero threshold doesn't filter anything."""
    embeddings = np.random.randn(100, 10)
    metadata = pd.DataFrame({
        'sample_code': [f"S{i:03d}" for i in range(100)],
        'tissue_pct': np.random.uniform(0, 100, 100),
        'has_invasion': np.random.randint(0, 2, 100),
        'invasion_pct': np.random.uniform(0, 50, 100),
    })
    
    filtered_embeddings, filtered_metadata = filter_by_tissue_percentage(
        embeddings, metadata, tissue_threshold=0.0
    )
    
    assert len(filtered_embeddings) == len(embeddings)
    assert len(filtered_metadata) == len(metadata)


def test_get_sample_groups():
    """Test extracting sample groups."""
    metadata = pd.DataFrame({
        'sample_code': ['A', 'A', 'B', 'B', 'C'],
        'tissue_pct': [50, 60, 70, 80, 90],
        'has_invasion': [0, 1, 0, 1, 1],
        'invasion_pct': [0, 10, 0, 20, 30],
    })
    
    groups = get_sample_groups(metadata)
    
    assert isinstance(groups, np.ndarray)
    assert len(groups) == 5
    assert list(groups) == ['A', 'A', 'B', 'B', 'C']


def test_get_labels():
    """Test extracting labels."""
    metadata = pd.DataFrame({
        'sample_code': ['A', 'B', 'C', 'D', 'E'],
        'tissue_pct': [50, 60, 70, 80, 90],
        'has_invasion': [0, 1, 0, 1, 1],
        'invasion_pct': [0, 10, 0, 20, 30],
    })
    
    labels = get_labels(metadata)
    
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 5
    assert labels.dtype == np.int64 or labels.dtype == np.int32
    assert list(labels) == [0, 1, 0, 1, 1]


def test_validate_data_for_cv_success():
    """Test validation passes with sufficient samples."""
    embeddings = np.random.randn(100, 10)
    metadata = pd.DataFrame({
        'sample_code': [f"S{i//10:02d}" for i in range(100)],  # 10 unique samples
        'tissue_pct': np.random.uniform(50, 100, 100),
        'has_invasion': np.random.randint(0, 2, 100),
        'invasion_pct': np.random.uniform(0, 50, 100),
    })
    
    # Should not raise with 5 folds and 10 samples
    validate_data_for_cv(embeddings, metadata, n_folds=5)


def test_validate_data_for_cv_failure():
    """Test validation fails with insufficient samples."""
    embeddings = np.random.randn(20, 10)
    metadata = pd.DataFrame({
        'sample_code': [f"S{i//10:02d}" for i in range(20)],  # Only 2 unique samples
        'tissue_pct': np.random.uniform(50, 100, 20),
        'has_invasion': np.random.randint(0, 2, 20),
        'invasion_pct': np.random.uniform(0, 50, 20),
    })
    
    # Should raise with 5 folds but only 2 samples
    with pytest.raises(ValueError, match="Not enough unique samples"):
        validate_data_for_cv(embeddings, metadata, n_folds=5)


def test_data_consistency():
    """Test that filtering maintains data consistency."""
    embeddings = np.random.randn(100, 10)
    metadata = pd.DataFrame({
        'sample_code': [f"S{i:03d}" for i in range(100)],
        'tissue_pct': np.arange(100),  # 0 to 99
        'has_invasion': np.random.randint(0, 2, 100),
        'invasion_pct': np.random.uniform(0, 50, 100),
    })
    
    # Filter with threshold 50 (should keep indices 50-99)
    filtered_embeddings, filtered_metadata = filter_by_tissue_percentage(
        embeddings, metadata, tissue_threshold=50.0
    )
    
    # Check that embeddings and metadata are aligned
    assert len(filtered_embeddings) == len(filtered_metadata)
    assert len(filtered_embeddings) == 50
    
    # Check that indices were reset
    assert filtered_metadata.index.tolist() == list(range(50))
