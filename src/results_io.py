"""
Results I/O utilities for saving and loading classification results.

This module provides functions to save cross-validation results to various
formats (CSV, JSON) and load them back for further analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

logger = logging.getLogger(__name__)


def save_results(
    fold_results: List[Dict[str, Any]],
    aggregated_metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    base_name: str,
) -> None:
    """
    Save cross-validation results to multiple formats.
    
    Saves:
        - Fold-level results to CSV (without confusion matrices)
        - Aggregated results to CSV
        - Complete results to JSON (with all metadata and confusion matrices)
    
    Args:
        fold_results: List of dictionaries with fold-level metrics
        aggregated_metrics: Dictionary with aggregated metrics
        config: Configuration dictionary
        output_dir: Directory to save results
        base_name: Base filename for output files
    
    Example:
        >>> save_results(fold_results, aggregated_metrics, config,
        ...              Path("results"), "cv_results_logistic_k5_tissue50")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fold-level results to CSV
    _save_fold_results_csv(fold_results, output_dir, base_name)
    
    # Save aggregated results to CSV
    _save_aggregated_results_csv(aggregated_metrics, output_dir, base_name)
    
    # Save complete results to JSON
    _save_complete_results_json(
        fold_results, aggregated_metrics, config, output_dir, base_name
    )


def _save_fold_results_csv(
    fold_results: List[Dict[str, Any]],
    output_dir: Path,
    base_name: str,
) -> None:
    """Save fold-level results to CSV."""
    csv_path = output_dir / f"{base_name}_folds.csv"
    
    # Prepare data for CSV (remove complex objects)
    fold_results_clean = []
    for fold in fold_results:
        fold_clean = fold.copy()
        # Remove items that are too complex for CSV
        fold_clean.pop("confusion_matrix", None)
        fold_clean.pop("y_proba", None)
        fold_results_clean.append(fold_clean)
    
    df_folds = pd.DataFrame(fold_results_clean)
    df_folds.to_csv(csv_path, index=False)
    logger.info(f"Saved fold results to {csv_path}")


def _save_aggregated_results_csv(
    aggregated_metrics: Dict[str, Any],
    output_dir: Path,
    base_name: str,
) -> None:
    """Save aggregated results to CSV."""
    agg_csv_path = output_dir / f"{base_name}_aggregated.csv"
    
    # Prepare data for CSV (remove confusion matrix)
    agg_clean = aggregated_metrics.copy()
    agg_clean.pop("confusion_matrix_total", None)
    
    df_agg = pd.DataFrame([agg_clean])
    df_agg.to_csv(agg_csv_path, index=False)
    logger.info(f"Saved aggregated results to {agg_csv_path}")


def _save_complete_results_json(
    fold_results: List[Dict[str, Any]],
    aggregated_metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    base_name: str,
) -> None:
    """Save complete results including all metadata to JSON."""
    json_path = output_dir / f"{base_name}_complete.json"
    
    # Prepare fold results for JSON (convert numpy arrays to lists)
    fold_results_json = []
    for fold in fold_results:
        fold_json = fold.copy()
        # Convert y_proba to list if present
        if fold_json.get("y_proba") is not None:
            import numpy as np
            if isinstance(fold_json["y_proba"], np.ndarray):
                fold_json["y_proba"] = fold_json["y_proba"].tolist()
        fold_results_json.append(fold_json)
    
    results = {
        "configuration": config,
        "fold_results": fold_results_json,
        "aggregated_metrics": aggregated_metrics,
    }
    
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved complete results to {json_path}")


def load_results(json_path: Path) -> Dict[str, Any]:
    """
    Load complete results from JSON file.
    
    Args:
        json_path: Path to JSON results file
    
    Returns:
        Dictionary with configuration, fold_results, and aggregated_metrics
    
    Example:
        >>> results = load_results(Path("results/cv_results_complete.json"))
        >>> print(results["configuration"])
        >>> print(f"Mean accuracy: {results['aggregated_metrics']['accuracy_mean']}")
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Results file not found: {json_path}")
    
    with open(json_path, "r") as f:
        results = json.load(f)
    
    logger.info(f"Loaded results from {json_path}")
    
    return results


def create_filename_base(
    classifier_type: str,
    n_folds: int,
    tissue_threshold: float,
) -> str:
    """
    Create a standardized base filename for results.
    
    Args:
        classifier_type: Type of classifier used
        n_folds: Number of folds
        tissue_threshold: Tissue percentage threshold
    
    Returns:
        Base filename string
    
    Example:
        >>> base_name = create_filename_base("logistic", 5, 50.0)
        >>> print(base_name)
        'cv_results_logistic_k5_tissue50'
    """
    return (
        f"cv_results_"
        f"{classifier_type}_"
        f"k{n_folds}_"
        f"tissue{tissue_threshold:.0f}"
    )


def export_to_excel(
    fold_results: List[Dict[str, Any]],
    aggregated_metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Export results to Excel file with multiple sheets.
    
    Args:
        fold_results: List of dictionaries with fold-level metrics
        aggregated_metrics: Dictionary with aggregated metrics
        output_path: Path to save Excel file
    
    Example:
        >>> export_to_excel(fold_results, aggregated_metrics,
        ...                 Path("results/cv_results.xlsx"))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare fold results
    fold_results_clean = []
    for fold in fold_results:
        fold_clean = fold.copy()
        fold_clean.pop("confusion_matrix", None)
        fold_clean.pop("y_proba", None)
        fold_results_clean.append(fold_clean)
    
    df_folds = pd.DataFrame(fold_results_clean)
    
    # Prepare aggregated results
    agg_clean = aggregated_metrics.copy()
    agg_clean.pop("confusion_matrix_total", None)
    df_agg = pd.DataFrame([agg_clean])
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_folds.to_excel(writer, sheet_name='Fold Results', index=False)
        df_agg.to_excel(writer, sheet_name='Aggregated Metrics', index=False)
    
    logger.info(f"Saved results to Excel: {output_path}")
