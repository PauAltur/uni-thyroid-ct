# Source Modules Documentation

This directory contains modular, reusable components for the thyroid CT classification pipeline.

## Module Overview

### 📁 `data_loader.py`
**Purpose**: Loading and filtering embeddings and metadata from HDF5 files.

**Key Functions**:
- `load_embeddings_from_h5()` - Load embeddings with optional tissue filtering
- `filter_by_tissue_percentage()` - Filter patches by tissue content
- `get_sample_groups()` - Extract sample identifiers for grouping
- `get_labels()` - Extract binary labels
- `validate_data_for_cv()` - Validate data is sufficient for K-fold CV

**Example**:
```python
from data_loader import load_embeddings_from_h5

embeddings, metadata = load_embeddings_from_h5(
    Path("embeddings.h5"),
    tissue_threshold=50.0
)
```

### 🤖 `models.py`
**Purpose**: Classifier factory for creating different types of classifiers.

**Key Functions**:
- `get_classifier()` - Create classifier instance with default hyperparameters
- `get_available_classifiers()` - List available classifier types

**Supported Classifiers**:
- Logistic Regression (`"logistic"`)
- Random Forest (`"random_forest"`)
- Gradient Boosting (`"gradient_boosting"`)
- Support Vector Machine (`"svm"`)

**Example**:
```python
from models import get_classifier

clf = get_classifier("logistic", random_state=42)
clf = get_classifier("random_forest", n_estimators=200, max_depth=15)
```

### 🔄 `cross_validation.py`
**Purpose**: Cross-validation with sample-level splitting.

**Key Functions**:
- `get_group_kfold_splits()` - Generate K-fold splits with grouping
- `train_and_evaluate_fold()` - Train and evaluate a single fold
- `run_cross_validation()` - Complete CV pipeline

**Features**:
- Uses `GroupKFold` to prevent data leakage
- All patches from same sample stay together
- Optional feature standardization
- Automatic leakage detection

**Example**:
```python
from cross_validation import run_cross_validation
from models import get_classifier

clf = get_classifier("logistic")
fold_meta, y_true_list, y_pred_list = run_cross_validation(
    X, y, groups, clf, n_folds=5, scale_features=True
)
```

### 📊 `metrics.py`
**Purpose**: Classification metrics computation and aggregation.

**Key Functions**:
- `compute_classification_metrics()` - Comprehensive metrics for one fold
- `aggregate_fold_metrics()` - Compute mean/std across folds
- `log_metrics()` - Pretty-print metrics
- `log_aggregated_metrics()` - Pretty-print aggregated results

**Metrics Computed**:
- Accuracy, Precision, Recall, Specificity, F1 Score
- AUC-ROC (if probabilities available)
- Confusion matrix (TN, FP, FN, TP)
- Class support

**Example**:
```python
from metrics import compute_classification_metrics, aggregate_fold_metrics

# Single fold
metrics = compute_classification_metrics(y_true, y_pred, y_proba)
print(f"F1 Score: {metrics['f1']:.3f}")

# Aggregate across folds
aggregated = aggregate_fold_metrics(fold_results)
print(f"Mean Accuracy: {aggregated['accuracy_mean']:.3f}")
```

### 📈 `visualization.py`
**Purpose**: Generate plots and visualizations for results.

**Key Functions**:
- `plot_cv_results()` - 4-panel comprehensive visualization
- `plot_roc_curves()` - ROC curves for each fold
- `create_results_summary_table()` - Formatted results DataFrame

**Visualizations**:
1. Metrics across folds (bar plot)
2. Aggregated metrics with error bars
3. Total confusion matrix (heatmap)
4. Metric distribution (box plot)

**Example**:
```python
from visualization import plot_cv_results, plot_roc_curves

plot_cv_results(fold_results, aggregated_metrics, 
                Path("results/plots.png"), n_folds=5)

plot_roc_curves(fold_metadata, y_true_list, 
                Path("results/roc_curves.png"))
```

### 💾 `results_io.py`
**Purpose**: Save and load classification results.

**Key Functions**:
- `save_results()` - Save to CSV and JSON
- `load_results()` - Load from JSON
- `create_filename_base()` - Standardized filenames
- `export_to_excel()` - Export to multi-sheet Excel file

**File Formats**:
- **CSV**: Fold-level and aggregated metrics (without complex objects)
- **JSON**: Complete results including confusion matrices
- **Excel**: Multi-sheet workbook (optional)

**Example**:
```python
from results_io import save_results, load_results

# Save
save_results(fold_results, aggregated_metrics, config,
             Path("results"), "cv_results_logistic_k5_tissue50")

# Load
results = load_results(Path("results/cv_results_complete.json"))
```

## Module Dependencies

```
data_loader.py
    ├── h5py
    ├── numpy
    └── pandas

models.py
    └── scikit-learn (classifiers)

cross_validation.py
    ├── numpy
    ├── scikit-learn (GroupKFold, StandardScaler)
    └── models.py (indirectly via classifier cloning)

metrics.py
    ├── numpy
    └── scikit-learn (metrics functions)

visualization.py
    ├── numpy
    ├── pandas
    ├── matplotlib
    └── seaborn

results_io.py
    ├── json
    ├── pandas
    └── pathlib
```

## Using the Modules

### Basic Workflow

```python
from pathlib import Path

# 1. Load data
from data_loader import load_embeddings_from_h5, get_labels, get_sample_groups
embeddings, metadata = load_embeddings_from_h5(Path("data.h5"), tissue_threshold=30)
X = embeddings
y = get_labels(metadata)
groups = get_sample_groups(metadata)

# 2. Create classifier
from models import get_classifier
clf = get_classifier("logistic", random_state=42)

# 3. Run cross-validation
from cross_validation import run_cross_validation
fold_meta, y_true_list, y_pred_list = run_cross_validation(
    X, y, groups, clf, n_folds=5
)

# 4. Compute metrics
from metrics import compute_classification_metrics, aggregate_fold_metrics
fold_results = []
for fold_meta, y_true, y_pred in zip(fold_meta, y_true_list, y_pred_list):
    metrics = compute_classification_metrics(y_true, y_pred, fold_meta.get("y_proba"))
    fold_results.append({**fold_meta, **metrics})

aggregated = aggregate_fold_metrics(fold_results)

# 5. Save and visualize
from results_io import save_results
from visualization import plot_cv_results

save_results(fold_results, aggregated, config, Path("results"), "my_experiment")
plot_cv_results(fold_results, aggregated, Path("results/plots.png"), n_folds=5)
```

### Custom Classifier

```python
from models import get_classifier

# Use custom hyperparameters
clf = get_classifier(
    "random_forest",
    random_state=42,
    n_estimators=500,  # Override default
    max_depth=20,      # Override default
    min_samples_split=5
)
```

### Reusing Components

All modules are standalone and can be used independently:

```python
# Just load data
from data_loader import load_embeddings_from_h5
embeddings, metadata = load_embeddings_from_h5(Path("data.h5"))

# Just compute metrics
from metrics import compute_classification_metrics
metrics = compute_classification_metrics(y_true, y_pred)

# Just create plots
from visualization import plot_cv_results
plot_cv_results(my_fold_results, my_aggregated_metrics, 
                Path("my_plot.png"), n_folds=10)
```

## Testing

Each module can be tested independently:

```bash
# Test data loading
python -c "from data_loader import load_embeddings_from_h5; \
           embeddings, metadata = load_embeddings_from_h5('test.h5'); \
           print(f'Loaded {len(embeddings)} embeddings')"

# Test classifier creation
python -c "from models import get_classifier, get_available_classifiers; \
           print(get_available_classifiers())"

# Test metrics
python -c "from metrics import compute_classification_metrics; \
           import numpy as np; \
           y_true = np.array([0, 0, 1, 1]); \
           y_pred = np.array([0, 1, 1, 1]); \
           print(compute_classification_metrics(y_true, y_pred))"
```

## Extension Points

### Adding a New Classifier

Edit `models.py`:

```python
# Add to DEFAULT_PARAMS
DEFAULT_PARAMS["my_classifier"] = {
    "param1": value1,
    "param2": value2,
}

# Add to get_classifier()
elif classifier_type == "my_classifier":
    from sklearn.xxx import MyClassifier
    clf = MyClassifier(**params)
```

### Adding a New Metric

Edit `metrics.py`:

```python
def compute_classification_metrics(...):
    # ... existing code ...
    
    # Add new metric
    metrics["my_metric"] = my_metric_function(y_true, y_pred)
    
    return metrics
```

### Adding a New Visualization

Edit `visualization.py`:

```python
def plot_my_custom_viz(fold_results, output_path):
    """Create custom visualization."""
    fig, ax = plt.subplots()
    # ... plotting code ...
    plt.savefig(output_path)
    plt.close()
```

## See Also

- Main script: `scripts/kfold_classification.py`
- Documentation: `scripts/README_CLASSIFICATION.md`
- Example notebook: `notebooks/05-kfold-classification-example.ipynb`
