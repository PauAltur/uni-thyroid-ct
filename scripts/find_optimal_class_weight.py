"""
Find optimal class weight for patch-level classification.

This script tests different class weight values to find the best trade-off
between precision, recall, and accuracy for imbalanced patch-level data.

Usage:
    python scripts/find_optimal_class_weight.py
"""

import sys
from pathlib import Path
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_embeddings_from_h5, prepare_classification_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def evaluate_class_weight(
    X_train, y_train, X_test, y_test,
    class_weight_ratio: float,
    random_state: int = 42
):
    """Train and evaluate model with specific class weight."""
    class_weight = {0: 1.0, 1: class_weight_ratio}
    
    clf = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        random_state=random_state,
        solver='lbfgs',
        C=1.0
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    
    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Get class 1 probabilities
    y_prob_pos = y_prob[:, 1]
    try:
        auc = roc_auc_score(y_test, y_prob_pos)
    except:  # noqa: E722
        auc = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Baseline accuracy (predict all class 0)
    baseline_acc = np.sum(y_test == 0) / len(y_test)
    
    return {
        'class_weight_ratio': class_weight_ratio,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'specificity': spec,
        'auc_roc': auc,
        'baseline_acc': baseline_acc,
        'acc_vs_baseline': acc - baseline_acc,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def main():
    logger.info("=" * 80)
    logger.info("Finding Optimal Class Weight for Patch-Level Classification")
    logger.info("=" * 80)
    
    # Configuration (adjust as needed)
    h5_path = Path("t:/users/altp/data/embeddings/thyroid_embeddings.h5")
    embeddings_key = "embeddings"
    label_column = "has_invasion"
    sample_column = "sample_codes"
    min_tissue_percentage = 0.01
    random_state = 42
    
    # Load data
    logger.info("\nLoading data...")
    embeddings, metadata = load_embeddings_from_h5(
        h5_path=h5_path,
        embeddings_key=embeddings_key,
        metadata_keys=None,
        min_tissue_percentage=min_tissue_percentage,
    )
    
    X, y, groups = prepare_classification_data(
        embeddings=embeddings,
        metadata=metadata,
        label_column=label_column,
        sample_column=sample_column,
    )
    
    # Use a manageable subset for faster testing
    # Sample 100k patches to speed up
    if len(X) > 100000:
        logger.info(f"\nSampling 100k patches from {len(X)} for faster testing...")
        np.random.seed(random_state)
        indices = np.random.choice(len(X), 100000, replace=False)
        X = X[indices]
        y = y[indices]
        groups = groups[indices]
    
    # Split ensuring groups don't leak
    unique_groups = np.unique(groups)
    np.random.seed(random_state)
    train_groups = np.random.choice(unique_groups, int(0.8 * len(unique_groups)), replace=False)
    
    train_mask = np.isin(groups, train_groups)
    test_mask = ~train_mask
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    logger.info(f"\nTrain set: {len(X_train)} patches, {np.bincount(y_train)}")
    logger.info(f"Test set:  {len(X_test)} patches, {np.bincount(y_test)}")
    
    # Test different class weight ratios
    logger.info("\n" + "=" * 80)
    logger.info("Testing Different Class Weights")
    logger.info("=" * 80)
    
    weight_ratios = [1, 2, 3, 5, 7, 10, 15, 18, 20, 25]
    results = []
    
    for ratio in weight_ratios:
        logger.info(f"\nTesting class_weight={{0: 1, 1: {ratio}}}...")
        result = evaluate_class_weight(X_train, y_train, X_test, y_test, ratio, random_state)
        results.append(result)
        
        logger.info(f"  Accuracy:    {result['accuracy']:.4f} (baseline: {result['baseline_acc']:.4f})")
        logger.info(f"  Precision:   {result['precision']:.4f}")
        logger.info(f"  Recall:      {result['recall']:.4f}")
        logger.info(f"  F1:          {result['f1']:.4f}")
        logger.info(f"  Specificity: {result['specificity']:.4f}")
        logger.info(f"  AUC-ROC:     {result['auc_roc']:.4f}")
        logger.info(f"  Confusion:   TP={result['tp']}, FP={result['fp']}, TN={result['tn']}, FN={result['fn']}")
    
    # Find best based on different criteria
    logger.info("\n" + "=" * 80)
    logger.info("Recommendations")
    logger.info("=" * 80)
    
    best_f1_idx = np.argmax([r['f1'] for r in results])
    best_auc_idx = np.argmax([r['auc_roc'] for r in results])
    best_acc_idx = np.argmax([r['accuracy'] for r in results])
    
    # Find balanced (F1 close to max, but accuracy > baseline)
    valid_results = [r for r in results if r['accuracy'] > r['baseline_acc']]
    if valid_results:
        best_balanced_idx = np.argmax([r['f1'] for r in valid_results])
        best_balanced = valid_results[best_balanced_idx]
    else:
        best_balanced = results[best_f1_idx]
    
    logger.info(f"\nBest F1 Score:     class_weight={{0: 1, 1: {results[best_f1_idx]['class_weight_ratio']}}}")
    logger.info(f"  F1={results[best_f1_idx]['f1']:.4f}, Acc={results[best_f1_idx]['accuracy']:.4f}")
    
    logger.info(f"\nBest AUC-ROC:      class_weight={{0: 1, 1: {results[best_auc_idx]['class_weight_ratio']}}}")
    logger.info(f"  AUC={results[best_auc_idx]['auc_roc']:.4f}, Acc={results[best_auc_idx]['accuracy']:.4f}")
    
    logger.info(f"\nBest Accuracy:     class_weight={{0: 1, 1: {results[best_acc_idx]['class_weight_ratio']}}}")
    logger.info(f"  Acc={results[best_acc_idx]['accuracy']:.4f}, F1={results[best_acc_idx]['f1']:.4f}")
    
    logger.info(f"\n🎯 RECOMMENDED (Balanced): class_weight={{0: 1, 1: {best_balanced['class_weight_ratio']}}}")
    logger.info(f"  Accuracy:    {best_balanced['accuracy']:.4f}")
    logger.info(f"  Precision:   {best_balanced['precision']:.4f}")
    logger.info(f"  Recall:      {best_balanced['recall']:.4f}")
    logger.info(f"  F1:          {best_balanced['f1']:.4f}")
    logger.info(f"  AUC-ROC:     {best_balanced['auc_roc']:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Update your config with the recommended class_weight:")
    logger.info(f"  class_weight: {{0: 1, 1: {best_balanced['class_weight_ratio']}}}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
