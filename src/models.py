"""
Model factory for creating different types of classifiers.

This module provides a centralized way to instantiate different classifier types
with sensible default hyperparameters.
"""

import logging
from typing import Dict, Any

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


# Default hyperparameters for each classifier type
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "logistic": {
        "max_iter": 1000,
        "n_jobs": -1,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "max_depth": 5,
    },
    "svm": {
        "kernel": "rbf",
        "probability": True,
    },
}


def get_classifier(
    classifier_type: str,
    random_state: int = 42,
    **kwargs
) -> BaseEstimator:
    """
    Get a classifier instance based on the specified type.
    
    Args:
        classifier_type: Type of classifier to create
            Options: "logistic", "random_forest", "gradient_boosting", "svm"
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters to pass to the classifier
            These will override default parameters
    
    Returns:
        Instantiated classifier
    
    Raises:
        ValueError: If classifier_type is not recognized
    
    Example:
        >>> clf = get_classifier("logistic", random_state=42)
        >>> clf = get_classifier("random_forest", n_estimators=200, max_depth=15)
    """
    classifier_type = classifier_type.lower()
    
    if classifier_type not in DEFAULT_PARAMS:
        raise ValueError(
            f"Unknown classifier type: {classifier_type}. "
            f"Choose from {list(DEFAULT_PARAMS.keys())}"
        )
    
    # Get default parameters and update with user-provided ones
    params = DEFAULT_PARAMS[classifier_type].copy()
    params.update(kwargs)
    params["random_state"] = random_state
    
    # Create classifier instance
    if classifier_type == "logistic":
        clf = LogisticRegression(**params)
    elif classifier_type == "random_forest":
        clf = RandomForestClassifier(**params)
    elif classifier_type == "gradient_boosting":
        clf = GradientBoostingClassifier(**params)
    elif classifier_type == "svm":
        clf = SVC(**params)
    
    logger.info(f"Created {classifier_type} classifier with parameters: {params}")
    
    return clf


def get_available_classifiers() -> list:
    """
    Get list of available classifier types.
    
    Returns:
        List of classifier type strings
    
    Example:
        >>> classifiers = get_available_classifiers()
        >>> print(f"Available: {classifiers}")
    """
    return list(DEFAULT_PARAMS.keys())
