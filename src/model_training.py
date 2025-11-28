"""
Model training utilities for classification tasks.

This module provides functions to train and evaluate various classifiers
on embedding data, with support for hyperparameter tuning and model persistence.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from typing import Dict, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def create_classifier(
    classifier_type: str = "logistic",
    random_state: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Create a classifier instance with specified hyperparameters.
    
    Args:
        classifier_type: Type of classifier ('logistic', 'random_forest', 'svm')
        random_state: Random state for reproducibility (if None, uses default 42)
        **kwargs: Hyperparameters to pass to the classifier
        
    Returns:
        Initialized classifier instance
    """
    if random_state is None:
        random_state = 42
    
    # Convert class_weight from DictConfig to dict if needed (for Hydra compatibility)
    if 'class_weight' in kwargs and kwargs['class_weight'] is not None:
        # Check if it's a DictConfig (from Hydra) and convert to regular dict
        if hasattr(kwargs['class_weight'], '__class__') and \
           kwargs['class_weight'].__class__.__name__ == 'DictConfig':
            # Convert to dict and ensure keys are integers for sklearn
            class_weight_dict = {}
            for k, v in kwargs['class_weight'].items():
                # Convert string keys to integers
                try:
                    key = int(k)
                except (ValueError, TypeError):
                    key = k
                class_weight_dict[key] = v
            kwargs['class_weight'] = class_weight_dict
    
    if classifier_type == "logistic":
        default_params = {
            "max_iter": 1000,
            "random_state": random_state,
            "solver": "lbfgs",
            "class_weight": "balanced",
        }
        default_params.update(kwargs)
        classifier = LogisticRegression(**default_params)
        logger.info(f"Created LogisticRegression with params: {default_params}")
        
    elif classifier_type == "random_forest":
        default_params = {
            "n_estimators": 100,
            "random_state": random_state,
            "max_depth": None,
            "class_weight": "balanced",
            "n_jobs": -1,
        }
        default_params.update(kwargs)
        classifier = RandomForestClassifier(**default_params)
        logger.info(f"Created RandomForestClassifier with params: {default_params}")
        
    elif classifier_type == "svm":
        default_params = {
            "kernel": "rbf",
            "random_state": random_state,
            "class_weight": "balanced",
            "probability": True,  # Enable probability estimates for AUC
        }
        default_params.update(kwargs)
        classifier = SVC(**default_params)
        logger.info(f"Created SVC with params: {default_params}")
        
    else:
        raise ValueError(
            f"Unknown classifier type: {classifier_type}. "
            f"Supported: 'logistic', 'random_forest', 'svm'"
        )
    
    return classifier


def train_classifier(
    classifier: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """
    Train a classifier on training data.
    
    Args:
        classifier: Initialized classifier instance
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained classifier
    """
    logger.info(f"Training {classifier.__class__.__name__}...")
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Feature dimension: {X_train.shape[1]}")
    logger.info(f"  Label distribution: {np.bincount(y_train)}")
    
    classifier.fit(X_train, y_train)
    
    logger.info("Training completed")
    return classifier


def predict_with_classifier(
    classifier: Any,
    X_test: np.ndarray,
    return_proba: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Make predictions with a trained classifier.
    
    Args:
        classifier: Trained classifier
        X_test: Test features
        return_proba: Whether to return class probabilities
        
    Returns:
        y_pred: Predicted labels
        y_prob: Predicted probabilities (if return_proba=True)
    """
    logger.info(f"Making predictions on {len(X_test)} samples...")
    
    # Predict labels
    y_pred = classifier.predict(X_test)
    
    # Predict probabilities if requested and supported
    y_prob = None
    if return_proba:
        if hasattr(classifier, "predict_proba"):
            y_prob = classifier.predict_proba(X_test)
        elif hasattr(classifier, "decision_function"):
            # For SVM without probability=True
            logger.warning("Classifier doesn't support predict_proba, using decision_function")
            y_prob = classifier.decision_function(X_test)
        else:
            logger.warning("Classifier doesn't support probability estimates")
    
    return y_pred, y_prob


def save_classifier(
    classifier: Any,
    save_path: Path,
) -> None:
    """
    Save a trained classifier to disk.
    
    Args:
        classifier: Trained classifier to save
        save_path: Path to save the classifier
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(classifier, f)
    
    logger.info(f"Saved classifier to {save_path}")


def load_classifier(
    load_path: Path,
) -> Any:
    """
    Load a trained classifier from disk.
    
    Args:
        load_path: Path to the saved classifier
        
    Returns:
        Loaded classifier instance
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"Classifier file not found: {load_path}")
    
    with open(load_path, "rb") as f:
        classifier = pickle.load(f)
    
    logger.info(f"Loaded classifier from {load_path}")
    return classifier


def train_and_evaluate_fold(
    classifier_type: str,
    classifier_params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: Optional[int] = None,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Train and evaluate a classifier on a single fold.
    
    Args:
        classifier_type: Type of classifier to use
        classifier_params: Hyperparameters for the classifier
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        random_state: Random state for reproducibility
        
    Returns:
        classifier: Trained classifier
        y_pred: Predictions on test set
        y_prob: Predicted probabilities on test set
    """
    # Create and train classifier
    classifier = create_classifier(classifier_type, random_state=random_state, **classifier_params)
    classifier = train_classifier(classifier, X_train, y_train)
    
    # Make predictions
    y_pred, y_prob = predict_with_classifier(classifier, X_test, return_proba=True)
    
    return classifier, y_pred, y_prob
