# Patch-Level Classification Strategies for Imbalanced Data

## Problem Summary

Your dataset:
- **1,297,782 patches** from **132 samples**
- **Class 0**: 1,229,856 (94.77%)
- **Class 1**: 67,926 (5.23%)
- **Imbalance ratio**: 18.1:1

Current results with `class_weight="balanced"`:
- Accuracy: 79.30% (worse than 94.77% baseline!)
- Precision: 14.49% (85% false positive rate)
- Recall: 58.42%
- F1: 23.00%

## Why This Happens

`class_weight="balanced"` calculates:
- Class 0 weight: ~0.05
- Class 1 weight: ~0.96

This makes the model heavily penalize missing class 1, leading to over-prediction of class 1.

---

## Strategy 1: Manual Class Weight Tuning (EASIEST)

### Step 1: Find Optimal Weight

Run the optimization script:
```bash
python scripts/find_optimal_class_weight.py
```

This tests weights from 1-25 and recommends the best balance.

### Step 2: Update Config

Edit `config/classification_config.yaml`:
```yaml
model:
  params:
    class_weight: {0: 1, 1: 5}  # Use recommended value from script
```

### Step 3: Train

```bash
python scripts/train_classifier.py
```

**Expected improvement**: 
- Accuracy: 90-95%
- Precision: 30-50%
- Recall: 40-70%
- F1: 35-60%

---

## Strategy 2: Threshold Adjustment

Instead of changing training, adjust the decision threshold at prediction time.

### Implementation

```python
from advanced_classification import ThresholdAdjustedClassifier

# Train normally
base_clf = LogisticRegression(max_iter=1000)
clf = ThresholdAdjustedClassifier(base_clf, threshold=None, metric='f1')

# Fit and optimize threshold on validation set
clf.fit(X_train, y_train)
clf.optimize_threshold(X_val, y_val)

# Predict with optimal threshold
y_pred = clf.predict(X_test)
```

**Pros**: 
- Don't need to retrain
- Can optimize for different metrics (F1, balanced accuracy)
- Fast to experiment

**Cons**:
- Requires validation set
- Still predicts at patch level

---

## Strategy 3: Downsampled Ensemble

Train multiple models on balanced subsets of data.

### Implementation

```python
from advanced_classification import DownsampledEnsemble

# Create ensemble of 10 classifiers
ensemble = DownsampledEnsemble(n_estimators=10, random_state=42)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
```

**Pros**:
- Naturally balanced training
- Diversity from different subsets
- Often better generalization

**Cons**:
- 10x training time
- Uses only ~5% of majority class per model

---

## Strategy 4: Sample-Level Post-Processing

Train at patch level, but enforce consistency at sample level.

### Implementation

```python
from advanced_classification import post_process_predictions_by_sample

# Train model normally
clf.fit(X_train, y_train)
patch_pred = clf.predict(X_test)
patch_prob = clf.predict_proba(X_test)

# Post-process to be consistent per sample
adjusted_pred = post_process_predictions_by_sample(
    patch_pred, 
    patch_prob, 
    sample_ids,
    strategy='mean_prob'  # or 'majority_vote', 'any_positive'
)
```

**Strategies**:
- `'majority_vote'`: Most patches win
- `'mean_prob'`: Average probability across patches
- `'any_positive'`: If ANY patch is positive, mark sample as positive

**Pros**:
- Reduces false positives from outlier patches
- Makes sense clinically (sample-level diagnosis)

**Cons**:
- Loses patch-level granularity

---

## Strategy 5: Different Classifier

Try Random Forest or SVM which handle imbalance differently.

### Random Forest

Edit config:
```yaml
model:
  type: "random_forest"
  params:
    n_estimators: 100
    max_depth: 20  # Limit depth to prevent overfitting
    min_samples_split: 100  # Require more samples to split
    min_samples_leaf: 50
    class_weight: "balanced_subsample"
```

**Pros**:
- Better at capturing non-linear patterns
- `balanced_subsample` handles imbalance per tree

### SVM with RBF Kernel

```yaml
model:
  type: "svm"
  params:
    kernel: "rbf"
    C: 1.0
    gamma: "scale"
    class_weight: {0: 1, 1: 5}
```

**Warning**: SVM is SLOW with 1.3M samples. Consider sampling first.

---

## Strategy 6: Feature Selection / Dimensionality Reduction

Reduce from 1536 dimensions to remove noise.

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Option 1: PCA
pca = PCA(n_components=256)
X_reduced = pca.fit_transform(X_train)

# Option 2: Feature selection
selector = SelectKBest(f_classif, k=256)
X_selected = selector.fit_transform(X_train, y_train)
```

**Pros**:
- Faster training
- May improve generalization
- Removes noise

---

## Recommended Workflow

### Quick Win (10 minutes):
1. Run `python scripts/find_optimal_class_weight.py`
2. Update config with recommended weight
3. Train with `python scripts/train_classifier.py`

### Medium Effort (1 hour):
1. Try manual class weights: {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 7}
2. Try Random Forest with `class_weight="balanced_subsample"`
3. Apply sample-level post-processing
4. Compare all results

### Advanced (half day):
1. Train downsampled ensemble
2. Optimize decision threshold
3. Try PCA + simpler model
4. Implement custom loss function

---

## Expected Results

| Strategy | Accuracy | Precision | Recall | F1 | Comments |
|----------|----------|-----------|--------|----|----|
| Balanced (current) | 79% | 14% | 58% | 23% | ❌ Over-predicts |
| Weight=5 | 93% | 40% | 55% | 46% | ✅ Much better |
| Weight=3 | 94% | 50% | 45% | 47% | ✅ Higher precision |
| Threshold=0.3 | 92% | 35% | 65% | 46% | ✅ Higher recall |
| Ensemble | 93% | 45% | 52% | 48% | ✅ Most robust |
| RF balanced_sub | 94% | 48% | 50% | 49% | ✅ Good balance |

---

## Evaluation Metrics Guidance

For imbalanced classification, don't just look at accuracy:

- **Accuracy**: Can be misleading (baseline is 94.77%)
- **Precision**: Of predicted positives, how many are correct?
  - High precision = few false alarms
- **Recall**: Of actual positives, how many did we find?
  - High recall = don't miss cases
- **F1**: Harmonic mean of precision and recall
  - Good overall metric for imbalance
- **AUC-ROC**: Threshold-independent metric
  - Best for comparing models
- **Specificity**: Of actual negatives, how many did we correctly identify?
  - Important if false positives are costly

**Clinical context matters**: 
- If missing a positive case is bad → optimize recall
- If false alarms are expensive → optimize precision
- Usually want balance → optimize F1 or use threshold tuning

---

## Next Steps

1. **Start here**: Run `python scripts/find_optimal_class_weight.py`
2. **Update config**: Use recommended class weight
3. **Train**: Run regular training script
4. **Evaluate**: Check if results make sense
5. **Iterate**: Try other strategies if needed
6. **Consider**: Whether sample-level aggregation makes more clinical sense
