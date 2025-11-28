"""
MLflow utilities for experiment tracking and logging.

This module provides functions to initialize MLflow, log parameters, metrics,
artifacts, and models for machine learning experiments.
"""

import mlflow
import mlflow.sklearn
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

logger = logging.getLogger(__name__)


def setup_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
) -> None:
    """
    Set up MLflow tracking.
    
    Args:
        experiment_name: Name of the MLflow experiment
        tracking_uri: URI for MLflow tracking server (default: local mlruns folder)
        artifact_location: Directory to store artifacts (optional)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    else:
        # Use local mlruns directory by default
        logger.info("Using local MLflow tracking (mlruns directory)")
    
    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            if artifact_location:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
            else:
                experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Error setting up MLflow experiment: {e}")
        raise


def start_mlflow_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> mlflow.ActiveRun:
    """
    Start a new MLflow run.
    
    Args:
        run_name: Name for the run
        tags: Dictionary of tags to add to the run
        
    Returns:
        Active MLflow run context
    """
    run = mlflow.start_run(run_name=run_name, tags=tags)
    logger.info(f"Started MLflow run: {run.info.run_id}")
    if run_name:
        logger.info(f"Run name: {run_name}")
    return run


def log_hydra_config(cfg: DictConfig) -> None:
    """
    Log Hydra configuration to MLflow.
    
    Args:
        cfg: Hydra configuration object
    """
    # Convert config to dictionary and flatten for MLflow params
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Flatten nested config
    flat_params = flatten_dict(config_dict)
    
    # Log parameters
    for key, value in flat_params.items():
        # MLflow params must be strings or numbers
        if value is not None and not isinstance(value, (list, dict)):
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Could not log parameter {key}: {e}")
    
    # Save full config as artifact
    config_yaml = OmegaConf.to_yaml(cfg)
    mlflow.log_text(config_yaml, "config.yaml")
    logger.info("Logged configuration to MLflow")


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        sep: Separator between nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def log_fold_metrics(fold_idx: int, metrics: Dict[str, Any]) -> None:
    """
    Log metrics from a single cross-validation fold.
    
    Args:
        fold_idx: Fold number (1-indexed)
        metrics: Dictionary of metrics
    """
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float, np.number)) and not np.isnan(metric_value):
            mlflow.log_metric(f"fold_{fold_idx}_{metric_name}", float(metric_value), step=fold_idx)


def log_aggregated_metrics(aggregated_metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Log aggregated cross-validation metrics (mean and std).
    
    Args:
        aggregated_metrics: Dictionary with 'mean' and 'std' subdictionaries
    """
    for metric_name, mean_value in aggregated_metrics.get("mean", {}).items():
        if mean_value is not None and not np.isnan(mean_value):
            mlflow.log_metric(f"cv_mean_{metric_name}", float(mean_value))
            
            # Log std if available
            std_value = aggregated_metrics.get("std", {}).get(metric_name)
            if std_value is not None and not np.isnan(std_value):
                mlflow.log_metric(f"cv_std_{metric_name}", float(std_value))
    
    logger.info("Logged aggregated CV metrics to MLflow")


def log_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List] = None,
    filename: str = "confusion_matrix.png"
) -> None:
    """
    Create and log a confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        filename: Name for the saved plot
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        mlflow.log_figure(fig, filename)
        plt.close(fig)
        logger.info(f"Logged confusion matrix plot: {filename}")
    except Exception as e:
        logger.warning(f"Could not log confusion matrix plot: {e}")


def log_cv_results_table(
    fold_metrics: List[Dict],
    filename: str = "cv_results.csv"
) -> None:
    """
    Log cross-validation results table as an artifact.
    
    Args:
        fold_metrics: List of metrics dictionaries from each fold
        filename: Name for the saved CSV file
    """
    try:
        df = pd.DataFrame(fold_metrics)
        mlflow.log_table(data=df, artifact_file=filename)
        logger.info(f"Logged CV results table: {filename}")
    except Exception as e:
        logger.warning(f"Could not log CV results table: {e}")


def log_model(model: Any, artifact_path: str = "model") -> None:
    """
    Log a trained model to MLflow.
    
    Args:
        model: Trained model object
        artifact_path: Path within the run's artifact directory
    """
    try:
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info(f"Logged model to MLflow: {artifact_path}")
    except Exception as e:
        logger.warning(f"Could not log model: {e}")


def log_artifact(local_path: Path, artifact_path: Optional[str] = None) -> None:
    """
    Log a file as an artifact.
    
    Args:
        local_path: Path to local file
        artifact_path: Optional subdirectory within artifacts
    """
    try:
        mlflow.log_artifact(str(local_path), artifact_path)
        logger.info(f"Logged artifact: {local_path.name}")
    except Exception as e:
        logger.warning(f"Could not log artifact {local_path}: {e}")


def log_predictions(
    predictions_list: List[Dict],
    filename: str = "predictions.csv"
) -> None:
    """
    Log predictions as a table artifact.
    
    Args:
        predictions_list: List of prediction dictionaries
        filename: Name for the saved CSV file
    """
    try:
        df = pd.DataFrame(predictions_list)
        mlflow.log_table(data=df, artifact_file=filename)
        logger.info(f"Logged predictions: {filename}")
    except Exception as e:
        logger.warning(f"Could not log predictions: {e}")


def log_data_statistics(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray
) -> None:
    """
    Log dataset statistics as metrics.
    
    Args:
        X: Feature matrix
        y: Labels
        groups: Group identifiers
    """
    try:
        mlflow.log_metric("n_samples", len(X))
        mlflow.log_metric("n_features", X.shape[1])
        mlflow.log_metric("n_unique_groups", len(np.unique(groups)))
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            mlflow.log_metric(f"n_class_{label}", int(count))
            mlflow.log_metric(f"pct_class_{label}", float(count / len(y) * 100))
        
        # Imbalance ratio
        if len(unique) == 2:
            imbalance_ratio = counts.max() / counts.min()
            mlflow.log_metric("imbalance_ratio", float(imbalance_ratio))
        
        logger.info("Logged data statistics to MLflow")
    except Exception as e:
        logger.warning(f"Could not log data statistics: {e}")


def end_mlflow_run(status: str = "FINISHED") -> None:
    """
    End the current MLflow run.
    
    Args:
        status: Run status ('FINISHED', 'FAILED', 'KILLED')
    """
    try:
        mlflow.end_run(status=status)
        logger.info(f"Ended MLflow run with status: {status}")
    except Exception as e:
        logger.warning(f"Error ending MLflow run: {e}")


def get_mlflow_ui_command(tracking_uri: Optional[str] = None) -> str:
    """
    Get the command to start MLflow UI.
    
    Args:
        tracking_uri: MLflow tracking URI (if using local, returns default UI command)
        
    Returns:
        Command string to start MLflow UI
    """
    if tracking_uri and not tracking_uri.startswith("file://"):
        # Remote tracking server - UI is likely already running
        return f"MLflow UI should be available at: {tracking_uri}"
    else:
        # Local tracking - provide command to start UI
        return "To view results in MLflow UI, run: mlflow ui"
