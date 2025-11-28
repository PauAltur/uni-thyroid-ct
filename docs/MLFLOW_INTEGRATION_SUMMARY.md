# MLflow Integration Summary

## What's New

MLflow experiment tracking has been integrated into the Thyroid CT Classification project. This allows you to:

- **Track all experiments** in a centralized location
- **Compare runs** side-by-side to find the best hyperparameters
- **Store models** with their metrics and parameters
- **Visualize results** in an interactive web interface
- **Share results** with team members via a local or remote server

## Quick Start

### 1. Install MLflow

```powershell
pip install mlflow
```

Or update all requirements:

```powershell
pip install -r requirements.txt
```

### 2. Run Training with MLflow Tracking

The MLflow integration is **enabled by default** in the configuration:

```powershell
# Patch-level classification
python scripts/train_classifier.py

# Sample-level classification
python scripts/train_classifier_sample_level.py

# Class weight optimization
python scripts/find_optimal_class_weight.py
```

### 3. View Results in MLflow UI

In a separate terminal, launch the MLflow UI:

```powershell
# Option 1: Use the convenience script
python scripts/launch_mlflow_ui.py

# Option 2: Direct command
mlflow ui
```

Then open your browser to: **http://127.0.0.1:5000**

## Files Modified/Added

### New Files
- `src/mlflow_utils.py` - MLflow utility functions for logging
- `scripts/launch_mlflow_ui.py` - Convenience script to launch UI
- `docs/MLFLOW_GUIDE.md` - Comprehensive MLflow usage guide
- `docs/MLFLOW_INTEGRATION_SUMMARY.md` - This file

### Modified Files
- `requirements.txt` - Added `mlflow` dependency
- `config/classification_config.yaml` - Added MLflow configuration section
- `scripts/train_classifier.py` - Integrated MLflow tracking
- `scripts/train_classifier_sample_level.py` - Integrated MLflow tracking
- `scripts/find_optimal_class_weight.py` - Integrated MLflow tracking

## Key Features

### What Gets Logged

1. **Configuration**: All Hydra config parameters
2. **Data Statistics**: Sample counts, class distributions, imbalance ratios
3. **Per-Fold Metrics**: Accuracy, precision, recall, F1, specificity, AUC-ROC for each CV fold
4. **Aggregated Metrics**: Mean ± std across all folds
5. **Confusion Matrices**: Visual plots of predictions
6. **Models**: Trained classifier (first fold as representative)
7. **Artifacts**: CSV results, predictions, configuration files

### Configuration Options

In `config/classification_config.yaml`:

```yaml
mlflow:
  # Enable/disable tracking
  enabled: true
  
  # Experiment name
  experiment_name: "thyroid_classification"
  
  # Tracking URI (null = local mlruns directory)
  # For local server: "http://127.0.0.1:5000"
  tracking_uri: null
  
  # What to log
  log_models: true
  log_confusion_matrix: true
  log_predictions: true
  log_artifacts: true
  
  # Custom tags
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
```

### Disable MLflow

To run without MLflow tracking:

```yaml
mlflow:
  enabled: false
```

Or from command line:

```powershell
python scripts/train_classifier.py mlflow.enabled=false
```

## Common Workflows

### Compare Different Class Weights

```powershell
# Run experiments with different weights
python scripts/train_classifier.py model.params.class_weight={0:1,1:5}
python scripts/train_classifier.py model.params.class_weight={0:1,1:10}
python scripts/train_classifier.py model.params.class_weight={0:1,1:15}

# Launch UI and compare
python scripts/launch_mlflow_ui.py
```

In the UI:
1. Select the runs you want to compare
2. Click "Compare"
3. View side-by-side metrics and parameters

### Find Best Model

1. Run multiple experiments
2. Open MLflow UI
3. Sort by `cv_mean_f1` or `cv_mean_accuracy`
4. Click on best run to see details
5. Download model artifact for deployment

### Track Class Weight Optimization

```powershell
python scripts/find_optimal_class_weight.py
```

This logs all tested weight ratios as metrics, making it easy to visualize which weight performed best.

## Using a Remote MLflow Server

For team collaboration:

### On the Server Machine

```powershell
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000
```

### On Client Machines

Update `config/classification_config.yaml`:

```yaml
mlflow:
  tracking_uri: "http://server-ip:5000"
```

All experiments will now be tracked on the shared server!

## Troubleshooting

### Port Already in Use

```powershell
# Use a different port
python scripts/launch_mlflow_ui.py --port 5001
```

### Can't Find mlruns Directory

This is normal if:
- No experiments have been run yet
- You're using a remote tracking server

Run a training script first to create experiments.

### Disable MLflow for Debugging

```powershell
python scripts/train_classifier.py mlflow.enabled=false
```

## Learn More

See the comprehensive guide: `docs/MLFLOW_GUIDE.md`

## Benefits

✅ **Centralized Tracking** - All experiments in one place  
✅ **Easy Comparison** - Compare hyperparameters and results visually  
✅ **Reproducibility** - Complete config saved with each run  
✅ **Collaboration** - Share results via local/remote server  
✅ **Model Management** - Store and version trained models  
✅ **Visualization** - Interactive plots and confusion matrices  
✅ **History** - Never lose experiment results again  

## Example Output

When you run a training script with MLflow enabled, you'll see:

```
================================================================================
Setting up MLflow Tracking
================================================================================
Created new MLflow experiment: thyroid_classification (ID: 1)
Started MLflow run: abc123def456
MLflow tracking initialized
View results: To view results in MLflow UI, run: mlflow ui
...
[Training proceeds normally]
...
View results in MLflow UI: To view results in MLflow UI, run: mlflow ui
Ended MLflow run with status: FINISHED
```

Navigate to http://127.0.0.1:5000 to explore the results!
