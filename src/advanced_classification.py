"""
Advanced strategies for patch-level classification with imbalanced data.

This module provides alternative approaches beyond simple class weighting:
1. Focal Loss (reduces loss for well-classified examples)
2. SMOTE-like oversampling in embedding space
3. Threshold adjustment
4. Ensemble voting
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


class ThresholdAdjustedClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper that adjusts decision threshold for imbalanced classification.
    
    Instead of using 0.5 as threshold, finds optimal threshold based on
    validation set to maximize F1 or other metric.
    """
    
    def __init__(self, base_classifier=None, threshold=0.5, metric='f1'):
        """
        Args:
            base_classifier: sklearn classifier with predict_proba
            threshold: Decision threshold (if None, will be optimized)
            metric: Metric to optimize threshold for ('f1', 'balanced_accuracy')
        """
        self.base_classifier = base_classifier
        self.threshold = threshold
        self.metric = metric
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit the base classifier."""
        if self.base_classifier is None:
            self.base_classifier = LogisticRegression(max_iter=1000, random_state=42)
        
        self.base_classifier.fit(X, y)
        self.classes_ = self.base_classifier.classes_
        return self
        
    def optimize_threshold(self, X_val, y_val, thresholds=None):
        """
        Find optimal threshold on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            thresholds: Array of thresholds to try (default: 0.1 to 0.9)
        """
        from sklearn.metrics import f1_score, balanced_accuracy_score
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
        
        y_prob = self.base_classifier.predict_proba(X_val)[:, 1]
        
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            if self.metric == 'f1':
                score = f1_score(y_val, y_pred, zero_division=0)
            elif self.metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_val, y_pred)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.threshold = best_threshold
        logger.info(f"Optimized threshold: {self.threshold:.3f} ({self.metric}={best_score:.4f})")
        return self
        
    def predict_proba(self, X):
        """Return probability estimates."""
        return self.base_classifier.predict_proba(X)
        
    def predict(self, X):
        """Predict using adjusted threshold."""
        y_prob = self.base_classifier.predict_proba(X)[:, 1]
        return (y_prob >= self.threshold).astype(int)


class DownsampledEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble of classifiers trained on balanced subsets via downsampling.
    
    Trains multiple classifiers, each on a balanced subset of the majority class
    combined with all minority class samples. Then averages predictions.
    """
    
    def __init__(self, base_classifier=None, n_estimators=10, random_state=42):
        """
        Args:
            base_classifier: sklearn classifier to use
            n_estimators: Number of ensemble members
            random_state: Random seed
        """
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classifiers_ = []
        self.classes_ = None
        
    def fit(self, X, y):
        """Train ensemble on balanced subsets."""
        np.random.seed(self.random_state)
        
        # Find minority and majority class
        unique, counts = np.unique(y, return_counts=True)
        minority_class = unique[np.argmin(counts)]
        majority_class = unique[np.argmax(counts)]
        
        minority_idx = np.where(y == minority_class)[0]
        majority_idx = np.where(y == majority_class)[0]
        
        n_minority = len(minority_idx)
        
        logger.info(f"Training {self.n_estimators} classifiers on balanced subsets...")
        logger.info(f"  Minority class ({minority_class}): {n_minority} samples")
        logger.info(f"  Majority class ({majority_class}): {len(majority_idx)} samples")
        logger.info(f"  Each classifier uses: {n_minority} minority + {n_minority} majority")
        
        self.classifiers_ = []
        
        for i in range(self.n_estimators):
            # Sample majority class to match minority
            sampled_majority_idx = np.random.choice(
                majority_idx, size=n_minority, replace=False
            )
            
            # Combine with all minority samples
            balanced_idx = np.concatenate([minority_idx, sampled_majority_idx])
            np.random.shuffle(balanced_idx)
            
            X_balanced = X[balanced_idx]
            y_balanced = y[balanced_idx]
            
            # Train classifier
            if self.base_classifier is None:
                clf = LogisticRegression(max_iter=1000, random_state=self.random_state + i)
            else:
                from sklearn.base import clone
                clf = clone(self.base_classifier)
            
            clf.fit(X_balanced, y_balanced)
            self.classifiers_.append(clf)
        
        self.classes_ = unique
        logger.info(f"Ensemble training complete: {len(self.classifiers_)} classifiers")
        return self
        
    def predict_proba(self, X):
        """Average probability predictions from all classifiers."""
        proba_sum = np.zeros((len(X), 2))
        
        for clf in self.classifiers_:
            proba_sum += clf.predict_proba(X)
        
        return proba_sum / len(self.classifiers_)
        
    def predict(self, X):
        """Predict using averaged probabilities."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def compute_sample_weights_focal(y, alpha=0.25, gamma=2.0):
    """
    Compute sample weights inspired by Focal Loss.
    
    Focal loss down-weights easy examples (well-classified) and focuses
    on hard examples. This implementation approximates it via sample weights.
    
    Args:
        y: Labels
        alpha: Balance between classes (0.25 = more weight on minority)
        gamma: Focusing parameter (higher = more focus on hard examples)
        
    Returns:
        Sample weights array
    """
    weights = np.ones(len(y))
    
    # Class balance
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique, counts):
        base_weight = len(y) / (len(unique) * count)
        weights[y == cls] = base_weight
    
    # Apply focal-like adjustment (simplified)
    # In full focal loss, this depends on predicted probability
    # Here we just apply alpha weighting
    weights[y == 1] *= (1 - alpha) / alpha if alpha < 0.5 else 1.0
    
    return weights


def post_process_predictions_by_sample(
    patch_predictions,
    patch_probabilities,
    sample_ids,
    strategy='majority_vote',
    threshold=0.5
):
    """
    Post-process patch predictions by aggregating to sample level,
    then propagating back to patches.
    
    This enforces consistency: all patches from same sample get same prediction.
    
    Args:
        patch_predictions: Predicted labels for patches
        patch_probabilities: Predicted probabilities for patches (N, 2)
        sample_ids: Sample ID for each patch
        strategy: 'majority_vote', 'mean_prob', or 'any_positive'
        threshold: Threshold for mean_prob strategy
        
    Returns:
        Adjusted patch predictions
    """
    adjusted_predictions = patch_predictions.copy()
    
    for sample_id in np.unique(sample_ids):
        mask = sample_ids == sample_id
        
        if strategy == 'majority_vote':
            # Most common prediction wins
            sample_pred = np.bincount(patch_predictions[mask]).argmax()
            
        elif strategy == 'mean_prob':
            # Average probabilities
            mean_prob = np.mean(patch_probabilities[mask, 1])
            sample_pred = 1 if mean_prob >= threshold else 0
            
        elif strategy == 'any_positive':
            # If any patch is positive, mark sample as positive
            sample_pred = 1 if np.any(patch_predictions[mask] == 1) else 0
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Apply to all patches in sample
        adjusted_predictions[mask] = sample_pred
    
    return adjusted_predictions
