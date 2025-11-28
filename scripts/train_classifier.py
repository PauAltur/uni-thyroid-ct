"""
Classification training pipeline with k-fold cross-validation.

This script trains classification models on embeddings extracted from CT volumes,
using group-aware cross-validation to prevent data leakage from samples with
multiple patches.

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py data.min_tissue_percentage=0.5
    python scripts/train_classifier.py model.type=random_forest cross_validation.n_folds=10
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import logging

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_embeddings_from_h5, prepare_classification_data
from cross_validation import create_group_kfold_splits
from metrics import compute_fold_metrics, aggregate_cv_metrics
from model_training import train_and_evaluate_fold, save_classifier
from mlflow_utils import (
    setup_mlflow, start_mlflow_run, end_mlflow_run,
    log_hydra_config, log_fold_metrics, log_aggregated_metrics,
    log_confusion_matrix_plot, log_cv_results_table, log_model,
    log_artifact, log_predictions, log_data_statistics,
    get_mlflow_ui_command
)

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> None:
    """
    Set up logging configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if cfg.logging.log_file:
        file_handler = logging.FileHandler(cfg.logging.log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)


def save_cv_results(
    fold_metrics: List[Dict],
    aggregated_metrics: Dict,
    output_path: Path,
) -> None:
    """
    Save cross-validation results to CSV.
    
    Args:
        fold_metrics: List of metrics from each fold
        aggregated_metrics: Aggregated statistics across folds
        output_path: Path to save the CSV file
    """
    # Convert fold metrics to DataFrame
    df = pd.DataFrame(fold_metrics)
    
    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved CV results to {output_path}")
    
    # Also save summary statistics
    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Cross-Validation Results Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Mean Metrics:\n")
        for metric, value in aggregated_metrics["mean"].items():
            if value is not None:
                std_value = aggregated_metrics["std"].get(metric, 0.0)
                f.write(f"  {metric}: {value:.4f} ± {std_value:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Per-Fold Results:\n\n")
        f.write(df.to_string())
    
    logger.info(f"Saved summary to {summary_path}")


def save_predictions(
    all_predictions: List[Dict],
    output_path: Path,
) -> None:
    """
    Save predictions from all folds to CSV.
    
    Args:
        all_predictions: List of prediction dictionaries
        output_path: Path to save the predictions CSV
    """
    df = pd.DataFrame(all_predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


@hydra.main(version_base=None, config_path="../config", config_name="classification_config")
def main(cfg: DictConfig) -> None:
    """
    Main classification training pipeline.
    
    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    setup_logging(cfg)
    
    logger.info("=" * 80)
    logger.info("Starting Classification Training Pipeline")
    logger.info("=" * 80)
    
    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Setup MLflow if enabled
    mlflow_enabled = cfg.get("mlflow", {}).get("enabled", False)
    if mlflow_enabled:
        logger.info("\n" + "=" * 80)
        logger.info("Setting up MLflow Tracking")
        logger.info("=" * 80)
        
        try:
            setup_mlflow(
                experiment_name=cfg.mlflow.experiment_name,
                tracking_uri=cfg.mlflow.get("tracking_uri"),
                artifact_location=cfg.mlflow.get("artifact_location"),
            )
            
            # Prepare tags
            tags = dict(cfg.mlflow.get("tags", {})) if cfg.mlflow.get("tags") else {}
            
            # Start MLflow run
            run_name = cfg.mlflow.get("run_name")
            if run_name is None:
                # Auto-generate run name from experiment and timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{cfg.paths.experiment_name}_{timestamp}"
            
            start_mlflow_run(run_name=run_name, tags=tags)
            
            # Log configuration
            log_hydra_config(cfg)
            
            logger.info("MLflow tracking initialized")
            logger.info(f"View results: {get_mlflow_ui_command(cfg.mlflow.get('tracking_uri'))}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            logger.warning("Continuing without MLflow tracking...")
            mlflow_enabled = False
    
    try:
        # Set random seed
        np.random.seed(cfg.runtime.random_seed)
        
        # Use consistent random_state throughout the pipeline
        random_state = cfg.runtime.random_seed
        
        # Load data
        logger.info("\n" + "=" * 80)
        logger.info("Loading Data")
        logger.info("=" * 80)
        
        embeddings, metadata = load_embeddings_from_h5(
            h5_path=Path(cfg.paths.embeddings_h5),
            embeddings_key=cfg.data.embeddings_key,
            metadata_keys=cfg.data.metadata_keys,
            min_tissue_percentage=cfg.data.min_tissue_percentage,
        )
        
        # Prepare data for classification
        X, y, groups = prepare_classification_data(
            embeddings=embeddings,
            metadata=metadata,
            label_column=cfg.data.label_column,
            sample_column=cfg.data.sample_column,
        )
        
        # Log data statistics to MLflow
        if mlflow_enabled:
            log_data_statistics(X, y, groups)
        
        # Cross-validation
        logger.info("\n" + "=" * 80)
        logger.info("Running Cross-Validation")
        logger.info("=" * 80)
        
        fold_metrics = []
        all_predictions = []
        trained_models = []
        all_y_true = []
        all_y_pred = []
        
        cv_splits = create_group_kfold_splits(
            X=X,
            y=y,
            groups=groups,
            n_folds=cfg.cross_validation.n_folds,
            shuffle=cfg.cross_validation.shuffle,
            random_state=random_state,
        )
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Fold {fold_idx}/{cfg.cross_validation.n_folds}")
            logger.info(f"{'=' * 80}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_test = groups[test_idx]
            
            # Train and evaluate
            classifier, y_pred, y_prob = train_and_evaluate_fold(
                classifier_type=cfg.model.type,
                classifier_params=dict(cfg.model.params),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                random_state=random_state,
            )
            
            # Compute metrics
            metrics = compute_fold_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                fold_idx=fold_idx,
                class_labels=classifier.classes_,
            )
            
            fold_metrics.append(metrics)
            
            # Log fold metrics to MLflow
            if mlflow_enabled:
                log_fold_metrics(fold_idx, metrics)
            
            # Accumulate predictions for overall confusion matrix
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            
            # Save predictions if requested
            if cfg.output.save_predictions:
                for i, (true_label, pred_label, sample) in enumerate(zip(y_test, y_pred, groups_test)):
                    pred_dict = {
                        "fold": fold_idx,
                        "sample": sample,
                        "true_label": int(true_label),
                        "predicted_label": int(pred_label),
                    }
                    if y_prob is not None:
                        if y_prob.ndim == 2:
                            # Save probabilities for all classes
                            for class_idx, class_label in enumerate(classifier.classes_):
                                pred_dict[f"prob_class_{class_label}"] = float(y_prob[i, class_idx])
                        else:
                            # Single probability value (e.g., from decision_function)
                            pred_dict["probability"] = float(y_prob[i])
                    all_predictions.append(pred_dict)
            
            # Save model if requested
            if cfg.output.save_models:
                trained_models.append(classifier)
        
        # Aggregate results
        logger.info("\n" + "=" * 80)
        logger.info("Aggregating Results")
        logger.info("=" * 80)
        
        aggregated_metrics = aggregate_cv_metrics(fold_metrics)
        
        # Log aggregated metrics to MLflow
        if mlflow_enabled:
            log_aggregated_metrics(aggregated_metrics)
            
            # Log overall confusion matrix
            if cfg.mlflow.get("log_confusion_matrix", True):
                log_confusion_matrix_plot(
                    np.array(all_y_true),
                    np.array(all_y_pred),
                    labels=[0, 1],
                    filename="overall_confusion_matrix.png"
                )
            
            # Log CV results table
            if cfg.mlflow.get("log_artifacts", True):
                log_cv_results_table(fold_metrics, "cv_results.csv")
            
            # Log predictions
            if cfg.mlflow.get("log_predictions", True) and all_predictions:
                log_predictions(all_predictions, "predictions.csv")
        
        # Save results to files
        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if cfg.output.save_cv_results:
            csv_path = output_dir / cfg.output.csv_filename
            save_cv_results(fold_metrics, aggregated_metrics, csv_path)
            
            # Log as MLflow artifact if enabled
            if mlflow_enabled and cfg.mlflow.get("log_artifacts", True):
                log_artifact(csv_path)
                summary_path = output_dir / f"{csv_path.stem}_summary.txt"
                if summary_path.exists():
                    log_artifact(summary_path)
        
        if cfg.output.save_predictions and all_predictions:
            pred_path = output_dir / cfg.output.predictions_filename
            save_predictions(all_predictions, pred_path)
        
        if cfg.output.save_models:
            models_dir = output_dir / cfg.output.models_dir
            models_dir.mkdir(parents=True, exist_ok=True)
            for i, model in enumerate(trained_models, 1):
                model_path = models_dir / f"model_fold_{i}.pkl"
                save_classifier(model, model_path)
            
            # Log first model to MLflow as representative
            if mlflow_enabled and cfg.mlflow.get("log_models", True) and trained_models:
                log_model(trained_models[0], "model")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("Training Completed!")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"Experiment: {cfg.paths.experiment_name}")
        
        # Print final metrics
        logger.info("\nFinal Cross-Validation Metrics (mean ± std):")
        for metric in ["accuracy", "precision", "recall", "f1", "specificity", "auc_roc"]:
            if metric in aggregated_metrics["mean"]:
                mean_val = aggregated_metrics["mean"][metric]
                std_val = aggregated_metrics["std"].get(metric, 0.0)
                if mean_val is not None:
                    logger.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        if mlflow_enabled:
            logger.info(f"\nView results in MLflow UI: {get_mlflow_ui_command(cfg.mlflow.get('tracking_uri'))}")
        
        # End MLflow run successfully
        if mlflow_enabled:
            end_mlflow_run(status="FINISHED")
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        if mlflow_enabled:
            end_mlflow_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
