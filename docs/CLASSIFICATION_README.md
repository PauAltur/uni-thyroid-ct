# Classification Pipeline

This directory contains a modular classification pipeline for training models on embeddings extracted from medical imaging data.

## Overview

The pipeline performs k-fold cross-validation with **group-aware splitting** to ensure that all embeddings from the same sample stay in the same fold, preventing data leakage.

## Features

- **Group-aware cross-validation**: Ensures samples with multiple patches are not split across train/test sets
- **Tissue filtering**: Filter embeddings by minimum tissue percentage
- **Multiple classifiers**: Logistic Regression, Random Forest, SVM
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1, Specificity, AUC-ROC, Confusion Matrix
- **Modular design**: Separated into reusable modules in `src/`
- **Hydra configuration**: Easy to modify hyperparameters via YAML or CLI

## Project Structure

```
src/
├── data_loader.py          # Load embeddings from H5 files with filtering
├── cross_validation.py     # Group-aware k-fold splitting
├── metrics.py              # Classification metrics computation
└── model_training.py       # Classifier training and evaluation

scripts/
└── train_classifier.py     # Main training pipeline script

config/
└── classification_config.yaml  # Hyperparameters and paths
```

## Quick Start

### 1. Basic Usage

Run with default settings:

```powershell
python scripts/train_classifier.py
```

### 2. Modify Configuration via CLI

Change tissue filtering threshold:

```powershell
python scripts/train_classifier.py data.min_tissue_percentage=0.5
```

Use a different classifier:

```powershell
python scripts/train_classifier.py model.type=random_forest
```

Change number of folds:

```powershell
python scripts/train_classifier.py cross_validation.n_folds=10
```

Multiple parameters at once:

```powershell
python scripts/train_classifier.py model.type=svm data.min_tissue_percentage=1.0 cross_validation.n_folds=3
```

### 3. Edit Configuration File

Modify `config/classification_config.yaml` to change defaults:

```yaml
paths:
  embeddings_h5: "path/to/your/embeddings.h5"
  output_dir: "path/to/output"
  
data:
  min_tissue_percentage: 0.1  # Filter patches with <0.1% tissue
  label_column: "has_invasion"  # Target variable
  
model:
  type: "logistic"  # or "random_forest", "svm"
  params:
    C: 1.0
    max_iter: 1000
```

## Input Data Format

The pipeline expects an HDF5 file with:

- **Embeddings dataset**: Array of shape `(N, embedding_dim)`
- **Metadata**: Either in a `metadata/` group or as top-level datasets
  - Required: `sample_code` (for grouping)
  - Required: Label column (e.g., `has_invasion`)
  - Optional: `tissue_percentage` (for filtering)

Example H5 structure:

```
embeddings.h5
├── embeddings                  # (N, 1536) array
├── sample_codes                # (N,) array of strings
└── metadata/
    ├── has_invasion           # (N,) boolean or int
    ├── tissue_percentage      # (N,) float (0-100)
    ├── invasion_type          # (N,) int
    └── ...
```

## Output

The pipeline generates:

1. **CV Results CSV** (`cv_results_*.csv`):
   - One row per fold
   - Columns: accuracy, precision, recall, f1, specificity, auc_roc, true_positives, false_positives, etc.

2. **Summary Text File** (`cv_results_*_summary.txt`):
   - Mean ± std for each metric
   - Per-fold detailed results

3. **Predictions CSV** (optional, `cv_predictions.csv`):
   - Per-sample predictions with probabilities
   - Includes fold number and sample identifier

4. **Trained Models** (optional, `models/model_fold_*.pkl`):
   - Saved sklearn classifier for each fold

## Available Classifiers

### Logistic Regression (default)

```yaml
model:
  type: "logistic"
  params:
    C: 1.0
    max_iter: 1000
    solver: "lbfgs"
    class_weight: "balanced"
```

### Random Forest

```yaml
model:
  type: "random_forest"
  params:
    n_estimators: 100
    max_depth: null
    class_weight: "balanced"
```

### SVM

```yaml
model:
  type: "svm"
  params:
    kernel: "rbf"
    C: 1.0
    probability: true
```

## Metrics Computed

- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Accuracy adjusted for class imbalance
- **Precision**: True positives / (true positives + false positives)
- **Recall (Sensitivity)**: True positives / (true positives + false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Specificity**: True negatives / (true negatives + false positives)
- **AUC-ROC**: Area under the ROC curve
- **Confusion Matrix**: True/False Positives/Negatives

## Examples

### Example 1: Train with high tissue threshold

```powershell
python scripts/train_classifier.py data.min_tissue_percentage=5.0
```

### Example 2: Random Forest with 10 folds

```powershell
python scripts/train_classifier.py model.type=random_forest cross_validation.n_folds=10
```

### Example 3: Save models and predictions

```yaml
# In config file:
output:
  save_models: true
  save_predictions: true
```

```powershell
python scripts/train_classifier.py
```

### Example 4: Custom experiment name

```powershell
python scripts/train_classifier.py paths.experiment_name=my_experiment
```

## Troubleshooting

### Error: "Cannot filter by tissue percentage"

The H5 file doesn't have a `tissue_percentage` field in metadata. Either:
- Set `min_tissue_percentage: 0.0` to disable filtering
- Add tissue_percentage to your H5 file during preprocessing

### Error: "Number of unique groups is less than n_folds"

You have fewer unique samples than folds. Reduce `cross_validation.n_folds`.

### Warning: "Overlapping groups between train and test"

This shouldn't happen with GroupKFold. Check that your `sample_code` column is correctly identifying unique samples.

## Advanced Usage

### Using Different Label Columns

To predict different targets:

```powershell
python scripts/train_classifier.py data.label_column=invasion_type
```

### Loading Specific Metadata Fields

```yaml
data:
  metadata_keys:
    - has_invasion
    - tissue_percentage
    - sample_code
    - position_d
```

### Custom Random Seed

```powershell
python scripts/train_classifier.py runtime.random_seed=123
```

## Integration with MLflow (Future)

For experiment tracking, you can extend the pipeline to log metrics to MLflow:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(cfg.model.params)
    mlflow.log_metrics(aggregated_metrics["mean"])
```
