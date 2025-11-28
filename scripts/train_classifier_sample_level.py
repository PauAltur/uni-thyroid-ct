"""
Sample-level classification training pipeline.

This script aggregates patch-level embeddings to sample-level before training,
which is more appropriate for highly imbalanced datasets where patches from
the same sample are highly correlated.

Usage:
    python scripts/train_classifier_sample_level.py
    python scripts/train_classifier_sample_level.py data.aggregation_method=median
"""

import sys
from pathlib import Path
from typing import Dict, List
import logging

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_embeddings_from_h5
from sample_aggregation import aggregate_embeddings_by_sample
from metrics import compute_fold_metrics, aggregate_cv_metrics
from model_training import train_and_evaluate_fold, save_classifier
from mlflow_utils import (
    setup_mlflow, start_mlflow_run, end_mlflow_run,
    log_hydra_config, log_fold_metrics, log_aggregated_metrics,
    log_confusion_matrix_plot, log_cv_results_table, log_model,
    log_artifact, log_data_statistics, get_mlflow_ui_command
)

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(cfg: DictConfig) -> None:
    """Set up logging configuration."""
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
    """Save cross-validation results to CSV."""
    df = pd.DataFrame(fold_metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved CV results to {output_path}")
    
    summary_path = output_path.parent / f"{output_path.stem}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Sample-Level Cross-Validation Results Summary\n")
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


@hydra.main(version_base=None, config_path="../config", config_name="classification_config")
def main(cfg: DictConfig) -> None:
    """Main sample-level classification training pipeline."""
    setup_logging(cfg)
    
    logger.info("=" * 80)
    logger.info("Starting Sample-Level Classification Training Pipeline")
    logger.info("=" * 80)
    
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
                experiment_name=cfg.mlflow.experiment_name + "_sample_level",
                tracking_uri=cfg.mlflow.get("tracking_uri"),
                artifact_location=cfg.mlflow.get("artifact_location"),
            )
            
            tags = dict(cfg.mlflow.get("tags", {})) if cfg.mlflow.get("tags") else {}
            tags["level"] = "sample"  # Mark as sample-level
            
            run_name = cfg.mlflow.get("run_name")
            if run_name is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"sample_{cfg.paths.experiment_name}_{timestamp}"
            
            start_mlflow_run(run_name=run_name, tags=tags)
            log_hydra_config(cfg)
            
            logger.info("MLflow tracking initialized")
            logger.info(f"View results: {get_mlflow_ui_command(cfg.mlflow.get('tracking_uri'))}")
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            logger.warning("Continuing without MLflow tracking...")
            mlflow_enabled = False
    
    try:
        np.random.seed(cfg.runtime.random_seed)
        random_state = cfg.runtime.random_seed
        
        # Load patch-level data
        logger.info("\n" + "=" * 80)
        logger.info("Loading Patch-Level Data")
        logger.info("=" * 80)
        
        embeddings, metadata = load_embeddings_from_h5(
            h5_path=Path(cfg.paths.embeddings_h5),
            embeddings_key=cfg.data.embeddings_key,
            metadata_keys=cfg.data.metadata_keys,
            min_tissue_percentage=cfg.data.min_tissue_percentage,
        )
        
        # Aggregate to sample level
        logger.info("\n" + "=" * 80)
        logger.info("Aggregating to Sample Level")
        logger.info("=" * 80)
        
        aggregation_method = cfg.data.get("aggregation_method", "mean")
        X, y, sample_ids = aggregate_embeddings_by_sample(
            embeddings=embeddings,
            metadata=metadata,
            sample_column=cfg.data.sample_column,
            label_column=cfg.data.label_column,
            aggregation_method=aggregation_method,
        )
        
        # Log data statistics to MLflow
        if mlflow_enabled:
            log_data_statistics(X, y, sample_ids)
        
        # Cross-validation (now sample-based, no need for GroupKFold)
        logger.info("\n" + "=" * 80)
        logger.info("Running Stratified Cross-Validation")
        logger.info("=" * 80)
        
        fold_metrics = []
        trained_models = []
        all_y_true = []
        all_y_pred = []
        
        # Use StratifiedKFold since we're now at sample level
        skf = StratifiedKFold(
            n_splits=cfg.cross_validation.n_folds,
            shuffle=cfg.cross_validation.shuffle,
            random_state=random_state,
        )
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Fold {fold_idx}/{cfg.cross_validation.n_folds}")
            logger.info(f"{'=' * 80}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Log fold statistics
            logger.info(f"Train: {len(X_train)} samples, label dist: {np.bincount(y_train)}")
            logger.info(f"Test:  {len(X_test)} samples, label dist: {np.bincount(y_test)}")
            
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
            
            if cfg.mlflow.get("log_confusion_matrix", True):
                log_confusion_matrix_plot(
                    np.array(all_y_true),
                    np.array(all_y_pred),
                    labels=[0, 1],
                    filename="sample_level_confusion_matrix.png"
                )
            
            if cfg.mlflow.get("log_artifacts", True):
                log_cv_results_table(fold_metrics, "sample_level_cv_results.csv")
        
        # Save results
        output_dir = Path(cfg.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if cfg.output.save_cv_results:
            csv_filename = cfg.output.csv_filename.replace(".csv", "_sample_level.csv")
            csv_path = output_dir / csv_filename
            save_cv_results(fold_metrics, aggregated_metrics, csv_path)
            
            if mlflow_enabled and cfg.mlflow.get("log_artifacts", True):
                log_artifact(csv_path)
                summary_path = output_dir / f"{csv_path.stem}_summary.txt"
                if summary_path.exists():
                    log_artifact(summary_path)
        
        if cfg.output.save_models:
            models_dir = output_dir / cfg.output.models_dir
            models_dir.mkdir(parents=True, exist_ok=True)
            for i, model in enumerate(trained_models, 1):
                model_path = models_dir / f"sample_model_fold_{i}.pkl"
                save_classifier(model, model_path)
            
            if mlflow_enabled and cfg.mlflow.get("log_models", True) and trained_models:
                log_model(trained_models[0], "sample_model")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("Training Completed!")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {output_dir}")
        
        logger.info("\nFinal Cross-Validation Metrics (mean ± std):")
        for metric in ["accuracy", "precision", "recall", "f1", "specificity", "auc_roc"]:
            if metric in aggregated_metrics["mean"]:
                mean_val = aggregated_metrics["mean"][metric]
                std_val = aggregated_metrics["std"].get(metric, 0.0)
                if mean_val is not None:
                    logger.info(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        if mlflow_enabled:
            logger.info(f"\nView results in MLflow UI: {get_mlflow_ui_command(cfg.mlflow.get('tracking_uri'))}")
        
        if mlflow_enabled:
            end_mlflow_run(status="FINISHED")
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        if mlflow_enabled:
            end_mlflow_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
