"""
Tests for models module.

Tests classifier factory functionality.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models import get_classifier, get_available_classifiers

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def test_get_classifier_logistic():
    """Test creating logistic regression classifier."""
    clf = get_classifier("logistic", random_state=42)
    
    assert isinstance(clf, LogisticRegression)
    assert clf.random_state == 42
    assert clf.max_iter == 1000
    assert clf.n_jobs == -1


def test_get_classifier_random_forest():
    """Test creating random forest classifier."""
    clf = get_classifier("random_forest", random_state=42)
    
    assert isinstance(clf, RandomForestClassifier)
    assert clf.random_state == 42
    assert clf.n_estimators == 100
    assert clf.max_depth == 10
    assert clf.n_jobs == -1


def test_get_classifier_gradient_boosting():
    """Test creating gradient boosting classifier."""
    clf = get_classifier("gradient_boosting", random_state=42)
    
    assert isinstance(clf, GradientBoostingClassifier)
    assert clf.random_state == 42
    assert clf.n_estimators == 100
    assert clf.max_depth == 5


def test_get_classifier_svm():
    """Test creating SVM classifier."""
    clf = get_classifier("svm", random_state=42)
    
    assert isinstance(clf, SVC)
    assert clf.random_state == 42
    assert clf.kernel == "rbf"
    assert clf.probability is True


def test_get_classifier_with_custom_params():
    """Test creating classifier with custom parameters."""
    clf = get_classifier(
        "random_forest",
        random_state=123,
        n_estimators=200,
        max_depth=15,
        min_samples_split=10
    )
    
    assert isinstance(clf, RandomForestClassifier)
    assert clf.random_state == 123
    assert clf.n_estimators == 200
    assert clf.max_depth == 15
    assert clf.min_samples_split == 10


def test_get_classifier_invalid_type():
    """Test error handling for invalid classifier type."""
    with pytest.raises(ValueError, match="Unknown classifier type"):
        get_classifier("invalid_classifier", random_state=42)


def test_get_classifier_case_insensitive():
    """Test that classifier type is case insensitive."""
    clf1 = get_classifier("LOGISTIC", random_state=42)
    clf2 = get_classifier("logistic", random_state=42)
    clf3 = get_classifier("Logistic", random_state=42)
    
    assert type(clf1) == type(clf2) == type(clf3)


def test_get_available_classifiers():
    """Test getting list of available classifiers."""
    classifiers = get_available_classifiers()
    
    assert isinstance(classifiers, list)
    assert len(classifiers) == 4
    assert "logistic" in classifiers
    assert "random_forest" in classifiers
    assert "gradient_boosting" in classifiers
    assert "svm" in classifiers


def test_classifier_default_params_consistency():
    """Test that default parameters are consistent."""
    # Create multiple instances with same parameters
    clf1 = get_classifier("logistic", random_state=42)
    clf2 = get_classifier("logistic", random_state=42)
    
    # Should have same parameters
    assert clf1.get_params() == clf2.get_params()


def test_random_state_propagation():
    """Test that random_state is properly set for all classifier types."""
    for clf_type in get_available_classifiers():
        clf = get_classifier(clf_type, random_state=999)
        assert clf.random_state == 999
