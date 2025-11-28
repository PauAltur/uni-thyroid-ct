# MLflow Configuration Examples

This file contains example MLflow configurations for different use cases.

## Default Configuration (Local Tracking)

```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification"
  tracking_uri: null  # Uses local mlruns directory
  artifact_location: null  # Default location
  run_name: null  # Auto-generated
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
  log_models: true
  log_confusion_matrix: true
  log_predictions: true
  log_artifacts: true
```

**Use case:** Single user, experiments stored locally in `mlruns/` directory.

## Local MLflow Server

```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification"
  tracking_uri: "http://127.0.0.1:5000"  # Local server
  artifact_location: null
  run_name: null
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
  log_models: true
  log_confusion_matrix: true
  log_predictions: true
  log_artifacts: true
```

**Setup:**
```powershell
# Start server in separate terminal
mlflow server --host 127.0.0.1 --port 5000
```

**Use case:** Better performance, can access while training is running, multiple terminal sessions.

## Remote MLflow Server (Team Collaboration)

```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification"
  tracking_uri: "http://192.168.1.100:5000"  # Replace with your server IP
  artifact_location: null
  run_name: null
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
    user: "${oc.env:USERNAME}"  # Automatically tag with username
  log_models: true
  log_confusion_matrix: true
  log_predictions: true
  log_artifacts: true
```

**Server setup:**
```powershell
# On the server machine
mlflow server --host 0.0.0.0 --port 5000
```

**Use case:** Team collaboration, centralized experiment tracking.

## Minimal Logging (Fast, Small Storage)

```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification"
  tracking_uri: null
  artifact_location: null
  run_name: null
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
  log_models: false  # Don't save models (save space)
  log_confusion_matrix: false  # Skip plots
  log_predictions: false  # Skip predictions CSV
  log_artifacts: false  # Skip artifact files
```

**Use case:** Quick experiments, limited disk space, only care about metrics.

## Detailed Logging with Custom Tags

```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification_ablation"
  tracking_uri: null
  artifact_location: null
  run_name: null
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
    phase: "ablation_study"
    dataset: "thyroid_ct"
    experiment_type: "hyperparameter_tuning"
    notes: "Testing different class weights"
    user: "researcher_name"
  log_models: true
  log_confusion_matrix: true
  log_predictions: true
  log_artifacts: true
```

**Use case:** Comprehensive experiment tracking with detailed organization.

## Disabled MLflow

```yaml
mlflow:
  enabled: false  # Turn off MLflow tracking
  experiment_name: "thyroid_classification"
  tracking_uri: null
  artifact_location: null
  run_name: null
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
  log_models: true
  log_confusion_matrix: true
  log_predictions: true
  log_artifacts: true
```

**Use case:** Debugging, quick tests, when MLflow server is unavailable.

## Custom Experiment Names by Use Case

### Hyperparameter Tuning
```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification_hyperparameter_tuning"
  # ... rest of config
```

### Class Weight Optimization
```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_class_weight_tuning"
  # ... rest of config
```

### Model Comparison
```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_model_comparison"
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
    comparison_group: "baseline_vs_tuned"
  # ... rest of config
```

### Final Production Run
```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification_production"
  run_name: "production_model_v1"
  tags:
    project: "thyroid-ct-classification"
    model_type: "${model.type}"
    stage: "production"
    version: "1.0"
  # ... rest of config
```

## Using Custom Artifact Locations

### Network Share (Windows)
```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification"
  tracking_uri: null
  artifact_location: "file:///Z:/shared/mlflow/artifacts"  # Network drive
  # ... rest of config
```

### Local Custom Path
```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification"
  tracking_uri: null
  artifact_location: "file:///T:/users/altp/mlflow_artifacts"  # Custom path
  # ... rest of config
```

## Command Line Overrides

You can override any MLflow setting from the command line:

```powershell
# Disable MLflow for a single run
python scripts/train_classifier.py mlflow.enabled=false

# Change experiment name
python scripts/train_classifier.py mlflow.experiment_name="test_experiment"

# Use remote server
python scripts/train_classifier.py mlflow.tracking_uri="http://server:5000"

# Set custom run name
python scripts/train_classifier.py mlflow.run_name="baseline_model"

# Add custom tags
python scripts/train_classifier.py mlflow.tags.phase="testing"

# Combine multiple overrides
python scripts/train_classifier.py \
    mlflow.experiment_name="ablation_study" \
    mlflow.run_name="test_1" \
    mlflow.tags.notes="Testing new feature"
```

## Tips for Organizing Experiments

### 1. Use Descriptive Experiment Names
```yaml
# Good
experiment_name: "thyroid_classification_tissue_threshold_study"

# Less clear
experiment_name: "experiment_1"
```

### 2. Add Meaningful Tags
```yaml
tags:
  project: "thyroid-ct-classification"
  model_type: "${model.type}"
  phase: "hyperparameter_tuning"
  tissue_threshold: "${data.min_tissue_percentage}"
  description: "Testing impact of tissue filtering"
```

### 3. Use Consistent Run Naming
```yaml
# Auto-generated with timestamp
run_name: null

# Or use descriptive names
run_name: "logistic_balanced_weights_v2"
```

### 4. Separate Experiments by Goal
- `thyroid_classification` - Main production experiments
- `thyroid_classification_dev` - Development/testing
- `thyroid_class_weight_tuning` - Specific optimization task
- `thyroid_ablation_study` - Feature importance studies

## Environment-Specific Configurations

### Development Environment
```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification_dev"
  tracking_uri: null  # Local
  log_models: false  # Save disk space
  log_predictions: false
```

### Production Environment
```yaml
mlflow:
  enabled: true
  experiment_name: "thyroid_classification_production"
  tracking_uri: "http://mlflow-server:5000"  # Remote
  log_models: true  # Save everything
  log_predictions: true
  log_artifacts: true
  tags:
    environment: "production"
    version: "1.0"
```

## Best Practices

1. **Keep MLflow enabled by default** - Small overhead, huge benefit
2. **Use descriptive experiment names** - Easy to find later
3. **Add tags for organization** - Filter and search effectively
4. **Start with local tracking** - Simplest setup
5. **Move to server when collaborating** - Share with team
6. **Regular cleanup** - Delete failed/test runs periodically
7. **Backup mlruns directory** - Don't lose experiment history

## Additional Examples

See the comprehensive guide for more details:
- [`docs/MLFLOW_GUIDE.md`](MLFLOW_GUIDE.md) - Full documentation
- [`docs/MLFLOW_INTEGRATION_SUMMARY.md`](MLFLOW_INTEGRATION_SUMMARY.md) - Quick start
- [`docs/MLFLOW_INSTALLATION_VERIFICATION.md`](MLFLOW_INSTALLATION_VERIFICATION.md) - Setup verification
