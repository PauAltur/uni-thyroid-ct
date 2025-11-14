"""
K-Fold Cross-Validation Classification Script for Patch-Level Invasion Detection

This script performs K-fold cross-validation on patch embeddings while ensuring
all patches from the same sample remain in the same fold (sample-level splitting).
It classifies patches as containing invasion or not, with configurable tissue
percentage filtering and comprehensive metrics computation.

Features:
- Sample-level K-fold splitting (patches from same sample stay together)
- Tissue percentage threshold filtering
- Multiple classification algorithms support
- Comprehensive metrics: accuracy, precision, recall, F1, AUC-ROC, confusion matrix
- Per-fold and aggregated results
- Results saved to CSV and JSON

Usage:
    python scripts/kfold_classification.py --h5_path path/to/embeddings.h5 --n_folds 5
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_embeddings_from_h5, get_sample_groups, get_labels, validate_data_for_cv
from models import get_classifier
from cross_validation import run_cross_validation
from metrics import compute_classification_metrics, aggregate_fold_metrics, log_metrics, log_aggregated_metrics
from visualization import plot_cv_results, plot_roc_curves
from results_io import save_results, create_filename_base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_classification_pipeline(
    h5_path: Path,
    n_folds: int,
    tissue_threshold: float,
    classifier_type: str,
    random_state: int,
    scale_features: bool,
    output_dir: Path,
) -> None:
    """
    Run the complete K-fold cross-validation classification pipeline.
    
    This function orchestrates all the modular components to:
    1. Load data and filter by tissue threshold
    2. Create classifier
    3. Run cross-validation
    4. Compute metrics
    5. Save results and generate plots
    
    Args:
        h5_path: Path to HDF5 file with embeddings
        n_folds: Number of cross-validation folds
        tissue_threshold: Minimum tissue percentage (0-100)
        classifier_type: Type of classifier to use
        random_state: Random seed for reproducibility
        scale_features: Whether to standardize features
        output_dir: Directory to save results
    """
    logger.info("="*60)
    logger.info("Starting Classification Pipeline")
    logger.info("="*60)
    logger.info("Configuration:")
    logger.info(f"  H5 Path: {h5_path}")
    logger.info(f"  Folds: {n_folds}")
    logger.info(f"  Tissue Threshold: {tissue_threshold}%")
    logger.info(f"  Classifier: {classifier_type}")
    logger.info(f"  Scale Features: {scale_features}")
    logger.info(f"  Output Dir: {output_dir}")
    
    # Step 1: Load data
    logger.info("\n" + "="*60)
    logger.info("Step 1: Loading Data")
    logger.info("="*60)
    embeddings, metadata = load_embeddings_from_h5(h5_path, tissue_threshold)
    
    # Step 2: Validate data for CV
    logger.info("\n" + "="*60)
    logger.info("Step 2: Validating Data")
    logger.info("="*60)
    validate_data_for_cv(embeddings, metadata, n_folds)
    
    # Step 3: Prepare data for training
    X = embeddings
    y = get_labels(metadata)
    groups = get_sample_groups(metadata)
    
    # Step 4: Create classifier
    logger.info("\n" + "="*60)
    logger.info("Step 4: Creating Classifier")
    logger.info("="*60)
    clf = get_classifier(classifier_type, random_state)
    
    # Step 5: Run cross-validation
    logger.info("\n" + "="*60)
    logger.info("Step 5: Running Cross-Validation")
    logger.info("="*60)
    fold_metadata, y_true_list, y_pred_list = run_cross_validation(
        X, y, groups, clf, n_folds, scale_features
    )
    
    # Step 6: Compute metrics for each fold
    logger.info("\n" + "="*60)
    logger.info("Step 6: Computing Metrics")
    logger.info("="*60)
    fold_results = []
    for i, (fold_meta, y_true, y_pred) in enumerate(zip(fold_metadata, y_true_list, y_pred_list), 1):
        metrics = compute_classification_metrics(y_true, y_pred, fold_meta.get("y_proba"))
        
        # Combine metadata and metrics
        fold_result = {
            **fold_meta,
            **metrics,
        }
        fold_results.append(fold_result)
        
        # Log fold metrics
        log_metrics(metrics, prefix=f"Fold {i} Results:")
    
    # Step 7: Aggregate metrics
    logger.info("\n" + "="*60)
    logger.info("Step 7: Aggregating Metrics")
    logger.info("="*60)
    aggregated_metrics = aggregate_fold_metrics(fold_results)
    log_aggregated_metrics(aggregated_metrics)
    
    # Step 8: Save results
    logger.info("\n" + "="*60)
    logger.info("Step 8: Saving Results")
    logger.info("="*60)
    config = {
        "h5_path": str(h5_path),
        "n_folds": n_folds,
        "tissue_threshold": tissue_threshold,
        "classifier_type": classifier_type,
        "random_state": random_state,
        "scale_features": scale_features,
    }
    base_name = create_filename_base(classifier_type, n_folds, tissue_threshold)
    save_results(fold_results, aggregated_metrics, config, output_dir, base_name)
    
    # Step 9: Generate visualizations
    logger.info("\n" + "="*60)
    logger.info("Step 9: Generating Visualizations")
    logger.info("="*60)
    plot_path = output_dir / f"{base_name}_plots.png"
    plot_cv_results(fold_results, aggregated_metrics, plot_path, n_folds)
    
    # Optional: Plot ROC curves if probabilities available
    if any(fold.get("y_proba") is not None for fold in fold_metadata):
        roc_path = output_dir / f"{base_name}_roc_curves.png"
        plot_roc_curves(fold_metadata, y_true_list, roc_path)
    
    logger.info("\n" + "="*60)
    logger.info("Classification pipeline completed successfully!")
    logger.info("="*60)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="K-Fold Cross-Validation for Patch-Level Invasion Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 5 folds
  python scripts/kfold_classification.py --h5_path embeddings/thyroid_embeddings.h5 --n_folds 5
  
  # With tissue filtering (only patches with >= 50% tissue)
  python scripts/kfold_classification.py --h5_path embeddings/thyroid_embeddings.h5 --n_folds 5 --tissue_threshold 50
  
  # Using Random Forest classifier
  python scripts/kfold_classification.py --h5_path embeddings/thyroid_embeddings.h5 --classifier random_forest
  
  # Multiple configurations
  python scripts/kfold_classification.py --h5_path embeddings/thyroid_embeddings.h5 --n_folds 10 --tissue_threshold 30 --classifier gradient_boosting --output_dir results/experiment1
        """
    )
    
    parser.add_argument(
        "--h5_path",
        type=str,
        required=True,
        help="Path to HDF5 file containing embeddings and metadata"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (default: 5)"
    )
    parser.add_argument(
        "--tissue_threshold",
        type=float,
        default=0.0,
        help="Minimum tissue percentage to include patch, 0-100 (default: 0.0, no filtering)"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest", "gradient_boosting", "svm"],
        help="Type of classifier to use (default: logistic)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no_scaling",
        action="store_true",
        help="Disable feature standardization (default: scaling enabled)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/classification",
        help="Directory to save results (default: results/classification)"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        run_classification_pipeline(
            h5_path=Path(args.h5_path),
            n_folds=args.n_folds,
            tissue_threshold=args.tissue_threshold,
            classifier_type=args.classifier,
            random_state=args.random_state,
            scale_features=not args.no_scaling,
            output_dir=Path(args.output_dir),
        )
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
