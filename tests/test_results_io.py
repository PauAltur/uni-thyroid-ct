"""
Tests for results_io module.

Tests saving and loading of classification results.
"""

import pytest
import json
import pandas as pd
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from results_io import (
    save_results,
    load_results,
    create_filename_base,
    export_to_excel,
)


@pytest.fixture
def sample_fold_results():
    """Create sample fold results for testing."""
    return [
        {
            'fold': 1,
            'train_size': 80,
            'test_size': 20,
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.90,
            'f1': 0.85,
            'confusion_matrix': [[8, 2], [1, 9]],
            'y_proba': [0.1, 0.9, 0.8, 0.2],
        },
        {
            'fold': 2,
            'train_size': 80,
            'test_size': 20,
            'accuracy': 0.90,
            'precision': 0.85,
            'recall': 0.95,
            'f1': 0.90,
            'confusion_matrix': [[9, 1], [0, 10]],
            'y_proba': [0.2, 0.95, 0.85, 0.15],
        },
    ]


@pytest.fixture
def sample_aggregated_metrics():
    """Create sample aggregated metrics for testing."""
    return {
        'accuracy_mean': 0.875,
        'accuracy_std': 0.025,
        'precision_mean': 0.825,
        'precision_std': 0.025,
        'confusion_matrix_total': [[17, 3], [1, 19]],
    }


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        'h5_path': '/path/to/data.h5',
        'n_folds': 5,
        'tissue_threshold': 50.0,
        'classifier_type': 'logistic',
        'random_state': 42,
        'scale_features': True,
    }


def test_save_results_creates_csv_files(sample_fold_results, sample_aggregated_metrics, sample_config):
    """Test that CSV files are created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = "test_results"
        
        save_results(
            sample_fold_results,
            sample_aggregated_metrics,
            sample_config,
            output_dir,
            base_name
        )
        
        # Check fold results CSV
        fold_csv = output_dir / f"{base_name}_folds.csv"
        assert fold_csv.exists()
        
        # Check aggregated results CSV
        agg_csv = output_dir / f"{base_name}_aggregated.csv"
        assert agg_csv.exists()


def test_save_results_creates_json_file(sample_fold_results, sample_aggregated_metrics, sample_config):
    """Test that JSON file is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = "test_results"
        
        save_results(
            sample_fold_results,
            sample_aggregated_metrics,
            sample_config,
            output_dir,
            base_name
        )
        
        # Check JSON file
        json_file = output_dir / f"{base_name}_complete.json"
        assert json_file.exists()


def test_save_results_csv_content(sample_fold_results, sample_aggregated_metrics, sample_config):
    """Test that CSV files contain correct data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = "test_results"
        
        save_results(
            sample_fold_results,
            sample_aggregated_metrics,
            sample_config,
            output_dir,
            base_name
        )
        
        # Read fold results CSV
        fold_csv = output_dir / f"{base_name}_folds.csv"
        df = pd.read_csv(fold_csv)
        
        assert len(df) == 2  # 2 folds
        assert 'fold' in df.columns
        assert 'accuracy' in df.columns
        assert 'confusion_matrix' not in df.columns  # Should be excluded from CSV


def test_save_results_json_content(sample_fold_results, sample_aggregated_metrics, sample_config):
    """Test that JSON file contains correct structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = "test_results"
        
        save_results(
            sample_fold_results,
            sample_aggregated_metrics,
            sample_config,
            output_dir,
            base_name
        )
        
        # Read JSON file
        json_file = output_dir / f"{base_name}_complete.json"
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        assert 'configuration' in data
        assert 'fold_results' in data
        assert 'aggregated_metrics' in data
        assert len(data['fold_results']) == 2


def test_save_results_creates_output_dir(sample_fold_results, sample_aggregated_metrics, sample_config):
    """Test that output directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "subdir" / "results"
        base_name = "test_results"
        
        save_results(
            sample_fold_results,
            sample_aggregated_metrics,
            sample_config,
            output_dir,
            base_name
        )
        
        assert output_dir.exists()
        assert (output_dir / f"{base_name}_folds.csv").exists()


def test_load_results(sample_fold_results, sample_aggregated_metrics, sample_config):
    """Test loading results from JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = "test_results"
        
        # Save results first
        save_results(
            sample_fold_results,
            sample_aggregated_metrics,
            sample_config,
            output_dir,
            base_name
        )
        
        # Load results
        json_file = output_dir / f"{base_name}_complete.json"
        loaded = load_results(json_file)
        
        assert 'configuration' in loaded
        assert 'fold_results' in loaded
        assert 'aggregated_metrics' in loaded
        assert loaded['configuration']['n_folds'] == 5


def test_load_results_file_not_found():
    """Test error handling when file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        load_results(Path("nonexistent.json"))


def test_create_filename_base():
    """Test filename generation."""
    base_name = create_filename_base("logistic", 5, 50.0)
    assert base_name == "cv_results_logistic_k5_tissue50"
    
    base_name = create_filename_base("random_forest", 10, 0.0)
    assert base_name == "cv_results_random_forest_k10_tissue0"
    
    base_name = create_filename_base("gradient_boosting", 3, 75.5)
    assert base_name == "cv_results_gradient_boosting_k3_tissue76"


def test_export_to_excel(sample_fold_results, sample_aggregated_metrics):
    """Test exporting to Excel file."""
    pytest.importorskip("openpyxl")  # Skip if openpyxl not installed
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.xlsx"
        
        export_to_excel(
            sample_fold_results,
            sample_aggregated_metrics,
            output_path
        )
        
        assert output_path.exists()


def test_export_to_excel_multiple_sheets(sample_fold_results, sample_aggregated_metrics):
    """Test that Excel file has multiple sheets."""
    pytest.importorskip("openpyxl")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "results.xlsx"
        
        export_to_excel(
            sample_fold_results,
            sample_aggregated_metrics,
            output_path
        )
        
        # Read Excel file
        df_folds = pd.read_excel(output_path, sheet_name='Fold Results')
        df_agg = pd.read_excel(output_path, sheet_name='Aggregated Metrics')
        
        assert len(df_folds) == 2  # 2 folds
        assert len(df_agg) == 1  # 1 aggregated row


def test_save_and_load_round_trip(sample_fold_results, sample_aggregated_metrics, sample_config):
    """Test that saving and loading preserves data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = "test_results"
        
        # Save
        save_results(
            sample_fold_results,
            sample_aggregated_metrics,
            sample_config,
            output_dir,
            base_name
        )
        
        # Load
        json_file = output_dir / f"{base_name}_complete.json"
        loaded = load_results(json_file)
        
        # Compare
        assert loaded['configuration'] == sample_config
        assert len(loaded['fold_results']) == len(sample_fold_results)
        assert loaded['aggregated_metrics']['accuracy_mean'] == sample_aggregated_metrics['accuracy_mean']


def test_save_results_handles_numpy_arrays(sample_aggregated_metrics, sample_config):
    """Test that numpy arrays in fold results are properly converted."""
    import numpy as np
    
    fold_results = [
        {
            'fold': 1,
            'accuracy': 0.85,
            'y_proba': np.array([0.1, 0.9, 0.8]),  # Numpy array
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        base_name = "test_results"
        
        # Should not raise an error
        save_results(
            fold_results,
            sample_aggregated_metrics,
            sample_config,
            output_dir,
            base_name
        )
        
        # Load and check
        json_file = output_dir / f"{base_name}_complete.json"
        loaded = load_results(json_file)
        
        # y_proba should be converted to list
        assert isinstance(loaded['fold_results'][0]['y_proba'], list)
