# K-Fold Classification Script

## Overview

The `kfold_classification.py` script performs K-fold cross-validation classification on patch embeddings to predict invasion. It ensures that **all patches from the same sample remain in the same fold** (sample-level splitting) to prevent data leakage and provide realistic performance estimates.

## Features

### Core Functionality
- ✅ **Sample-level K-fold splitting** - Uses `GroupKFold` to keep patches from the same sample together
- ✅ **Tissue percentage filtering** - Filter patches based on minimum tissue content threshold
- ✅ **Multiple classifiers** - Logistic Regression, Random Forest, Gradient Boosting, SVM
- ✅ **Feature standardization** - Optional z-score normalization of embeddings
- ✅ **Comprehensive metrics** - Accuracy, Precision, Recall, Specificity, F1, AUC-ROC

### Output
- 📊 **Fold-level results** (CSV)
- 📈 **Aggregated metrics** with mean ± std (CSV)
- 📋 **Complete results** including confusion matrices (JSON)
- 📉 **Visualization plots** - metrics across folds, confusion matrix, box plots

## Installation

The script requires the following packages (already in `requirements.txt`):

```bash
pip install scikit-learn h5py pandas numpy matplotlib seaborn
```

## Usage

### Basic Command Line Usage

```bash
# Basic 5-fold cross-validation with logistic regression
python scripts/kfold_classification.py \
    --h5_path path/to/embeddings.h5 \
    --n_folds 5
```

### Common Use Cases

#### 1. Filter by Tissue Percentage
Only include patches with at least 50% tissue content:

```bash
python scripts/kfold_classification.py \
    --h5_path embeddings/thyroid_embeddings.h5 \
    --n_folds 5 \
    --tissue_threshold 50
```

#### 2. Use Different Classifier
Try Random Forest instead of Logistic Regression:

```bash
python scripts/kfold_classification.py \
    --h5_path embeddings/thyroid_embeddings.h5 \
    --n_folds 5 \
    --classifier random_forest
```

#### 3. Custom Output Directory
Save results to a specific directory:

```bash
python scripts/kfold_classification.py \
    --h5_path embeddings/thyroid_embeddings.h5 \
    --n_folds 10 \
    --tissue_threshold 30 \
    --classifier gradient_boosting \
    --output_dir results/experiment1
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--h5_path` | str | **required** | Path to HDF5 file containing embeddings |
| `--n_folds` | int | 5 | Number of folds for cross-validation |
| `--tissue_threshold` | float | 0.0 | Minimum tissue percentage (0-100) |
| `--classifier` | str | logistic | Classifier type: `logistic`, `random_forest`, `gradient_boosting`, `svm` |
| `--random_state` | int | 42 | Random seed for reproducibility |
| `--no_scaling` | flag | False | Disable feature standardization |
| `--output_dir` | str | results/classification | Output directory for results |

### Programmatic Usage

You can also use the `PatchClassifier` class directly in Python:

```python
from pathlib import Path
from scripts.kfold_classification import PatchClassifier

# Create classifier instance
classifier = PatchClassifier(
    h5_path="embeddings/thyroid_embeddings.h5",
    n_folds=5,
    tissue_threshold=50.0,
    classifier_type="logistic",
    random_state=42,
    scale_features=True
)

# Run the full pipeline
classifier.run(output_dir=Path("results/my_experiment"))

# Access results directly
print("Aggregated Metrics:")
for key, value in classifier.aggregated_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
```

## Output Files

The script generates the following files in the output directory:

### 1. Fold-level Results CSV
`cv_results_{classifier}_k{n_folds}_tissue{threshold}_folds.csv`

Contains metrics for each individual fold:
- fold number
- train/test sizes
- train/test sample counts
- accuracy, precision, recall, specificity, f1, auc_roc
- true positives/negatives, false positives/negatives

### 2. Aggregated Results CSV
`cv_results_{classifier}_k{n_folds}_tissue{threshold}_aggregated.csv`

Contains mean ± std for each metric across all folds.

### 3. Complete Results JSON
`cv_results_{classifier}_k{n_folds}_tissue{threshold}_complete.json`

Contains:
- Configuration parameters
- Full fold-level results (including confusion matrices)
- Aggregated metrics

### 4. Visualization Plots PNG
`cv_results_{classifier}_k{n_folds}_tissue{threshold}_plots.png`

Four-panel visualization:
- Metrics across folds (bar chart)
- Aggregated metrics with error bars
- Total confusion matrix (heatmap)
- Metric distribution across folds (box plot)

## H5 File Structure Requirements

The script expects an HDF5 file with the following structure:

```
embeddings.h5
├── embeddings                      # shape: (n_patches, embedding_dim)
└── metadata/
    ├── sample_code                 # sample identifier for each patch
    ├── tissue_percentage           # tissue content (0-100)
    ├── has_invasion                # binary label (0 or 1)
    └── invasion_percentage         # invasion content (0-100)
```

## Classification Metrics

The script computes the following metrics for each fold and aggregated:

- **Accuracy**: Overall correctness (TP + TN) / Total
- **Precision**: Positive predictive value TP / (TP + FP)
- **Recall (Sensitivity)**: True positive rate TP / (TP + FN)
- **Specificity**: True negative rate TN / (TN + FP)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (requires probability predictions)
- **Confusion Matrix**: True/False positives/negatives

## Sample-Level Splitting

**Why it matters**: In medical imaging, multiple patches are extracted from each patient/sample. If patches from the same sample appear in both training and test sets, the model can "cheat" by learning sample-specific patterns, leading to unrealistically high performance.

**How it works**: This script uses `GroupKFold` from scikit-learn, which ensures that all patches from a given sample ID stay together in the same fold. This provides a more realistic estimate of how the model will perform on completely unseen patients.

## Examples

### Example 1: Quick Test
Test with default settings on a small dataset:

```bash
python scripts/kfold_classification.py \
    --h5_path embeddings/test_embeddings.h5 \
    --n_folds 3
```

### Example 2: Production Run
Full analysis with tissue filtering and Random Forest:

```bash
python scripts/kfold_classification.py \
    --h5_path embeddings/production_embeddings.h5 \
    --n_folds 10 \
    --tissue_threshold 40 \
    --classifier random_forest \
    --output_dir results/production/rf_tissue40
```

### Example 3: Comparing Classifiers
Run multiple experiments to compare classifiers:

```bash
for classifier in logistic random_forest gradient_boosting; do
    python scripts/kfold_classification.py \
        --h5_path embeddings/thyroid_embeddings.h5 \
        --n_folds 5 \
        --tissue_threshold 50 \
        --classifier $classifier \
        --output_dir results/comparison/$classifier
done
```

### Example 4: Grid Search Over Tissue Thresholds
Find optimal tissue threshold:

```bash
for threshold in 0 20 40 60 80; do
    python scripts/kfold_classification.py \
        --h5_path embeddings/thyroid_embeddings.h5 \
        --n_folds 5 \
        --tissue_threshold $threshold \
        --output_dir results/tissue_sweep/threshold_$threshold
done
```

## Troubleshooting

### Issue: "Not enough unique samples for K folds"
**Cause**: After tissue filtering, not enough unique samples remain for the requested number of folds.

**Solution**: 
- Reduce `n_folds` (e.g., from 10 to 5)
- Lower `tissue_threshold` to retain more samples
- Check data with: `python -c "import h5py; f=h5py.File('file.h5'); print(len(set(f['metadata/sample_code'][:])))"

### Issue: "AUC-ROC could not be computed"
**Cause**: One or more test folds contain only one class.

**Solution**:
- This can happen with highly imbalanced data
- Check class balance in your dataset
- Consider stratified sampling or different fold configuration
- The script continues and reports other metrics

### Issue: Memory Error
**Cause**: Loading very large embedding files into memory.

**Solution**:
- Process in batches (requires script modification)
- Use a subset of data for initial experiments
- Increase system RAM or use a machine with more memory

## Performance Notes

- **Logistic Regression**: Fastest, good baseline, works well with high-dimensional embeddings
- **Random Forest**: Slower, handles non-linear patterns, more robust to outliers
- **Gradient Boosting**: Slowest, often best performance, requires tuning
- **SVM**: Can be very slow with large datasets, good for smaller datasets

For large datasets (>100k patches), start with Logistic Regression for quick iteration.

## See Also

- [05-kfold-classification-example.ipynb](../notebooks/05-kfold-classification-example.ipynb) - Interactive examples
- [03-analyze-embeddings.ipynb](../notebooks/03-analyze-embeddings.ipynb) - Exploratory analysis
- Main project [README.md](../README.md)
