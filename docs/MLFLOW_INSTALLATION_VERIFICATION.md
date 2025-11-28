# MLflow Integration - Installation and Verification Guide

## Installation Steps

### 1. Install MLflow

The simplest way is to update all dependencies:

```powershell
pip install -r requirements.txt
```

Or install just MLflow:

```powershell
pip install mlflow
```

### 2. Verify Installation

Check that MLflow is installed correctly:

```powershell
mlflow --version
```

You should see output like: `mlflow, version 2.x.x`

## Quick Verification

### Test 1: Run a Training Script with MLflow

```powershell
# Run training with default settings (MLflow enabled by default)
python scripts/train_classifier.py
```

You should see output indicating MLflow tracking is active:
```
================================================================================
Setting up MLflow Tracking
================================================================================
Created new MLflow experiment: thyroid_classification (ID: 1)
Started MLflow run: abc123...
MLflow tracking initialized
View results: To view results in MLflow UI, run: mlflow ui
```

### Test 2: Launch MLflow UI

After running a training script, start the MLflow UI:

```powershell
# Option 1: Use the convenience script
python scripts/launch_mlflow_ui.py

# Option 2: Use the batch file (Windows)
launch_mlflow_ui.bat

# Option 3: Direct command
mlflow ui
```

The UI should start and automatically open in your browser at: http://127.0.0.1:5000

### Test 3: View Results

In the MLflow UI, you should see:

1. **Experiments** - Your `thyroid_classification` experiment
2. **Runs** - The training run you just completed
3. **Metrics** - Click on a run to see metrics like:
   - `cv_mean_accuracy`
   - `cv_mean_f1`
   - `fold_1_accuracy`, etc.
4. **Parameters** - All configuration settings
5. **Artifacts** - Files like:
   - `config.yaml`
   - `cv_results.csv`
   - `overall_confusion_matrix.png`

## Test the Class Weight Optimization Script

```powershell
python scripts/find_optimal_class_weight.py
```

This script will:
1. Test multiple class weight ratios
2. Log each result to MLflow
3. Recommend the best configuration
4. Save recommendation as an MLflow artifact

View the results in MLflow UI to see all tested weights.

## Configuration Test

### Test Enabling/Disabling MLflow

**Run with MLflow enabled (default):**
```powershell
python scripts/train_classifier.py
```

**Run with MLflow disabled:**
```powershell
python scripts/train_classifier.py mlflow.enabled=false
```

The second run should not create any MLflow entries.

### Test Remote Tracking

If you have a remote MLflow server:

1. **Start a local MLflow server** (in a separate terminal):
   ```powershell
   mlflow server --host 127.0.0.1 --port 5000
   ```

2. **Update config** to use the server:
   
   Edit `config/classification_config.yaml`:
   ```yaml
   mlflow:
     tracking_uri: "http://127.0.0.1:5000"
   ```

3. **Run training:**
   ```powershell
   python scripts/train_classifier.py
   ```

4. **Verify** in the MLflow UI (already running at http://127.0.0.1:5000)

## Troubleshooting

### Issue: MLflow command not found

**Solution:**
```powershell
# Ensure MLflow is installed
pip install mlflow

# Check if it's in PATH
where mlflow

# If not, try:
python -m mlflow --version
```

### Issue: Port 5000 already in use

**Solution:**
```powershell
# Use a different port
python scripts/launch_mlflow_ui.py --port 5001

# Or with direct command
mlflow ui --port 5001
```

### Issue: No runs appearing in UI

**Possible causes:**
1. MLflow was disabled in config
2. Training script encountered an error
3. Wrong `tracking_uri` configured

**Solution:**
- Check that `mlflow.enabled: true` in config
- Look for errors in training output
- Verify `mlruns` directory exists in project root
- If using remote server, check `tracking_uri` is correct

### Issue: Permission errors on Windows

**Solution:**
- Ensure you have write permissions to project directory
- Check Windows Firewall isn't blocking port 5000
- Run terminal as administrator if needed

## Files Created by MLflow

After running with MLflow enabled, you should see:

```
your-project/
├── mlruns/                          # MLflow tracking data
│   ├── 0/                           # Default experiment
│   └── 1/                           # Your experiments
│       ├── <run_id>/                # Individual run data
│       │   ├── artifacts/           # Saved files
│       │   │   ├── config.yaml
│       │   │   ├── cv_results.csv
│       │   │   └── overall_confusion_matrix.png
│       │   ├── metrics/             # Logged metrics
│       │   ├── params/              # Logged parameters
│       │   └── tags/                # Run tags
│       └── meta.yaml                # Experiment metadata
```

## Integration Points

The following scripts now include MLflow tracking:

1. ✅ `scripts/train_classifier.py` - Main patch-level training
2. ✅ `scripts/train_classifier_sample_level.py` - Sample-level training
3. ✅ `scripts/find_optimal_class_weight.py` - Class weight optimization

All scripts:
- Log configuration parameters
- Log training metrics
- Save confusion matrices
- Store artifacts (results, predictions)
- Support enabling/disabling via config

## Next Steps

1. **Run some experiments** with different hyperparameters:
   ```powershell
   python scripts/train_classifier.py model.params.class_weight={0:1,1:5}
   python scripts/train_classifier.py model.params.class_weight={0:1,1:10}
   python scripts/train_classifier.py model.params.class_weight={0:1,1:15}
   ```

2. **Compare results** in MLflow UI:
   - Select multiple runs
   - Click "Compare"
   - View side-by-side metrics

3. **Find best model**:
   - Sort runs by `cv_mean_f1` or `cv_mean_accuracy`
   - Download best model artifact

4. **Share results**:
   - Set up remote MLflow server for team collaboration
   - Or share `mlruns` directory

## Additional Resources

- **[docs/MLFLOW_GUIDE.md](../docs/MLFLOW_GUIDE.md)** - Comprehensive usage guide
- **[docs/MLFLOW_INTEGRATION_SUMMARY.md](../docs/MLFLOW_INTEGRATION_SUMMARY.md)** - Quick reference
- **[MLflow Documentation](https://mlflow.org/docs/latest/index.html)** - Official docs

## Success Checklist

- [ ] MLflow installed and version verified
- [ ] Training script runs with MLflow tracking
- [ ] MLflow UI launches successfully
- [ ] Can view experiments and runs in UI
- [ ] Metrics, parameters, and artifacts are logged
- [ ] Can compare multiple runs
- [ ] Understand how to enable/disable MLflow
- [ ] Know how to change tracking URI for remote server

If all items are checked, your MLflow integration is working correctly! 🎉
