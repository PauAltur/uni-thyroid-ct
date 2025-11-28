# MLflow Integration Guide

This guide explains how to use MLflow for experiment tracking in the Thyroid CT Classification project.

## Overview

MLflow is integrated into the training pipeline to track:
- Hyperparameters and configuration
- Training metrics (accuracy, precision, recall, F1, AUC-ROC, etc.)
- Cross-validation results
- Confusion matrices
- Trained models
- Predictions and artifacts

## Setup

### Install MLflow

MLflow is included in `requirements.txt`. Install it with:

```powershell
pip install mlflow
```

Or install all requirements:

```powershell
pip install -r requirements.txt
```

### Configuration

MLflow settings are configured in `config/classification_config.yaml`:

```yaml
mlflow:
  # Enable/disable MLflow tracking
  enabled: true
  
  # Experiment name in MLflow
  experiment_name: "thyroid_classification"
  
  # Tracking URI (null = local mlruns directory)
  # For local server: "http://127.0.0.1:5000"
  tracking_uri: null
  
  # Additional tags for runs
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
  
  # What to log
  log_models: true
  log_confusion_matrix: true
  log_predictions: true
  log_artifacts: true
```

## Using MLflow

### Option 1: Local Tracking (Default)

By default, MLflow stores experiments locally in the `mlruns` directory.

1. **Run your training script:**
   ```powershell
   python scripts/train_classifier.py
   ```

2. **View results in MLflow UI:**
   ```powershell
   mlflow ui
   ```

3. **Open your browser to:** http://127.0.0.1:5000

### Option 2: Local MLflow Server

For better performance and concurrent access:

1. **Start MLflow server** (in a separate terminal):
   ```powershell
   mlflow server --host 127.0.0.1 --port 5000
   ```

2. **Update config** to use the server:
   ```yaml
   mlflow:
     tracking_uri: "http://127.0.0.1:5000"
   ```

3. **Run training:**
   ```powershell
   python scripts/train_classifier.py
   ```

4. **View in browser:** http://127.0.0.1:5000

### Option 3: Remote MLflow Server

For team collaboration:

1. **Set up a remote MLflow server** (on a shared machine)

2. **Update config** with remote URI:
   ```yaml
   mlflow:
     tracking_uri: "http://your-server-ip:5000"
   ```

3. **Run training** - results automatically sync to the server

## What Gets Logged

### Parameters
- All configuration from `classification_config.yaml`
- Model hyperparameters
- Cross-validation settings
- Data processing parameters

### Metrics
- Per-fold metrics:
  - `fold_N_accuracy`
  - `fold_N_precision`
  - `fold_N_recall`
  - `fold_N_f1`
  - `fold_N_specificity`
  - `fold_N_auc_roc`

- Aggregated metrics:
  - `cv_mean_accuracy`
  - `cv_std_accuracy`
  - (and similar for other metrics)

### Artifacts
- `config.yaml` - Full experiment configuration
- `cv_results.csv` - Cross-validation results table
- `predictions.csv` - Model predictions (if enabled)
- `overall_confusion_matrix.png` - Confusion matrix plot
- `model/` - Trained model (first fold as representative)

### Tags
- `project` - Project identifier
- `model_type` - Type of classifier used
- Custom tags can be added in config

## Comparing Experiments

In the MLflow UI:

1. **Select multiple runs** using checkboxes
2. **Click "Compare"** button
3. **View side-by-side comparison** of:
   - Parameters
   - Metrics
   - Plots

You can also:
- Sort by metrics to find best runs
- Filter by parameters
- Search by tags
- Plot metric trends

## Example Workflows

### Hyperparameter Tuning

Compare different class weights:

```powershell
# Run with different class weights
python scripts/train_classifier.py model.params.class_weight={0:1,1:5}
python scripts/train_classifier.py model.params.class_weight={0:1,1:10}
python scripts/train_classifier.py model.params.class_weight={0:1,1:15}
```

Then compare in MLflow UI to find the best configuration.

### Finding Optimal Class Weight

The `find_optimal_class_weight.py` script includes MLflow tracking:

```powershell
python scripts/find_optimal_class_weight.py
```

This logs all tested weight ratios as metrics, making it easy to:
- See which weight performed best
- Compare metrics across different ratios
- Review recommendations

### Cross-Validation Analysis

After training, view in MLflow:
1. Per-fold variability in metrics
2. Overall confusion matrix
3. Predictions for error analysis

## Best Practices

### Naming Runs
Set meaningful run names in config:
```yaml
mlflow:
  run_name: "logistic_balanced_weights_v2"
```

Or auto-generate from experiment name (default).

### Organizing Experiments
Create separate experiments for different tasks:
- `thyroid_classification` - Main classification experiments
- `thyroid_class_weight_tuning` - Class weight optimization
- `thyroid_ablation_study` - Feature importance studies

Change in config:
```yaml
mlflow:
  experiment_name: "thyroid_ablation_study"
```

### Using Tags
Add tags for filtering and organization:
```yaml
mlflow:
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
    phase: "hyperparameter_tuning"
    tissue_threshold: "${data.min_tissue_percentage}"
```

### Storing Large Artifacts
For large datasets or many models:
1. Consider using remote artifact storage (S3, Azure, GCS)
2. Set `log_models: false` if models are very large
3. Use `log_predictions: false` for huge prediction sets

## Troubleshooting

### MLflow Not Starting
```powershell
# Check if port is in use
netstat -ano | findstr :5000

# Use different port
mlflow ui --port 5001
```

### Runs Not Appearing
- Check `mlruns` directory exists
- Verify `tracking_uri` is correct
- Look for errors in training logs

### Permission Issues
On Windows, ensure you have write permissions to:
- `mlruns/` directory
- Output directories specified in config

### Disable MLflow
To run without MLflow tracking:
```yaml
mlflow:
  enabled: false
```

## Advanced Features

### Nested Runs
For more complex experiments, use nested runs to group related experiments.

### Model Registry
Register best models for deployment:
1. Train and evaluate models
2. In MLflow UI, select best run
3. Click "Register Model"
4. Assign version and stage (Staging/Production)

### API Access
Query experiments programmatically:
```python
import mlflow

# Search runs
runs = mlflow.search_runs(
    experiment_names=["thyroid_classification"],
    filter_string="metrics.cv_mean_f1 > 0.8"
)

# Load best model
best_run_id = runs.iloc[0].run_id
model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")
```

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Python API](https://mlflow.org/docs/latest/python_api/index.html)
