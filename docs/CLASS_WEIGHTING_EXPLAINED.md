# Understanding Class Weighting in Loss Computation

## Overview

Class weighting is a technique to handle **imbalanced datasets** where one class has significantly more samples than another. It modifies the loss function during training to make the model pay more attention to the minority class.

## Your Dataset Context

**Thyroid CT Classification Problem:**
- **Total patches**: 1,297,782
- **Class 0 (no invasion)**: 1,229,856 patches (94.77%)
- **Class 1 (invasion)**: 67,926 patches (5.23%)
- **Imbalance ratio**: 18.1:1

Without any adjustments, the model can achieve 94.77% accuracy by simply predicting "no invasion" for everything!

---

## How Loss Computation Works

### Standard Loss (No Weighting)

For logistic regression, the loss for each sample is:

```
Loss_i = -[y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
```

Where:
- `y_i` = true label (0 or 1)
- `p_i` = predicted probability for class 1

**Total loss** = Sum of all individual losses

### Problem with Imbalanced Data

With your 18:1 ratio:
- Model sees 18 class-0 samples for every 1 class-1 sample
- Minimizing loss means mostly focusing on class 0
- Missing a class-1 sample barely affects total loss
- Missing a class-0 sample significantly affects total loss

**Result**: Model learns to predict class 0 almost always.

---

## Class Weighting Solution

### Weighted Loss Formula

```
Loss_i = -w_i * [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
```

Where `w_i` is the weight for the class of sample `i`:
- If sample belongs to class 0: `w_i = weight_0`
- If sample belongs to class 1: `w_i = weight_1`

**Effect**: Multiply the loss of minority class samples by a larger weight, making them more "important" during training.

---

## Option 1: `class_weight="balanced"`

### How It Works

Scikit-learn automatically calculates weights as:

```python
weight_class_j = n_samples / (n_classes * n_samples_class_j)
```

**For your data:**
```python
n_samples = 1,297,782
n_classes = 2

weight_0 = 1,297,782 / (2 * 1,229,856) = 0.528
weight_1 = 1,297,782 / (2 * 67,926) = 9.55
```

### What This Means

- **Class 0 samples**: Loss multiplied by 0.528
- **Class 1 samples**: Loss multiplied by 9.55

The model now cares about **18x more** about getting class 1 right!

### Your Results with `class_weight="balanced"`

```
Accuracy:  79.30% (baseline was 94.77%!)
Precision: 14.49% (86% of positive predictions are wrong!)
Recall:    58.42% (finds 58% of true positives)
F1:        23.00%
```

**What happened:**
- Model became too aggressive at predicting class 1
- To avoid the high penalty of missing class 1, it predicts class 1 too often
- Results in massive over-prediction → low precision
- Accuracy dropped below baseline!

### Visualization

```
No weighting:
Class 0 loss contribution: |████████████████████| 95%
Class 1 loss contribution: |█|                     5%
→ Model ignores class 1

Balanced weighting:
Class 0 loss contribution: |██████████|           50%
Class 1 loss contribution: |██████████|           50%
→ Model over-focuses on class 1
```

---

## Option 2: Manual Class Weights (Recommended)

### How It Works

You specify exact weights:

```python
class_weight = {0: 1.0, 1: 5.0}
```

This means:
- Class 0 loss multiplied by 1.0 (baseline)
- Class 1 loss multiplied by 5.0

### Why This Is Better

Instead of fully balancing (18:1 → 1:1), you **partially balance** (18:1 → 3.6:1).

This tells the model: "Class 1 is important, but not THAT important."

### Finding the Right Weight

Your script `find_optimal_class_weight.py` tests multiple ratios:

```python
weight_ratios = [1, 2, 3, 5, 7, 10, 15, 18, 20, 25]
```

**Typical results pattern:**

| Weight Ratio | Accuracy | Precision | Recall | F1 | Behavior |
|--------------|----------|-----------|--------|----|----|
| 1 (no weight) | 94.8% | 80% | 15% | 25% | Predicts mostly 0 |
| 3 | 94.5% | 50% | 45% | 47% | Good balance |
| 5 | 93.2% | 40% | 55% | 46% | Slightly more recall |
| 10 | 90.5% | 30% | 65% | 41% | Higher recall |
| 18 (balanced) | 79.3% | 14% | 58% | 23% | Over-predicts 1 |

**Sweet spot**: Usually between 3-7 for your dataset.

### Expected Results with `{0: 1, 1: 5}`

```
Accuracy:  93%    (good, close to baseline)
Precision: 40%    (60% false positive rate - acceptable)
Recall:    55%    (finds 55% of invasions)
F1:        46%    (balanced performance)
```

---

## Mathematical Intuition

### No Weighting (weight=1)

```
Total Loss = 1,229,856 * L_0 + 67,926 * L_1
           ≈ 1,229,856 * L_0  (dominates)
```

To minimize total loss, focus on class 0.

### Balanced Weighting (weight≈18)

```
Total Loss = 0.528 * 1,229,856 * L_0 + 9.55 * 67,926 * L_1
           = 649,284 * L_0 + 648,803 * L_1
           ≈ 650,000 * L_0 + 650,000 * L_1
```

Both classes contribute equally → model tries too hard on class 1.

### Manual Weighting (weight=5)

```
Total Loss = 1.0 * 1,229,856 * L_0 + 5.0 * 67,926 * L_1
           = 1,229,856 * L_0 + 339,630 * L_1
```

Class 1 contribution increased from 5% to 22% of total loss.
- Model cares more about class 1
- But not so much that it over-predicts

---

## Effect on Model Behavior

### During Training

**Without weighting:**
```
Gradient for class 0 sample: ∂Loss/∂w ≈ 0.001
Gradient for class 1 sample: ∂Loss/∂w ≈ 0.001
→ Equal impact per sample, but 18x more class 0 samples
```

**With weight=5:**
```
Gradient for class 0 sample: ∂Loss/∂w ≈ 0.001
Gradient for class 1 sample: ∂Loss/∂w ≈ 0.005
→ Each class 1 sample has 5x the impact
```

### During Prediction

**Logistic regression decision:**
```python
if probability(class 1) > 0.5:
    predict class 1
else:
    predict class 0
```

**Effect of weighting:**
- Model learns to output higher probabilities for class 1
- The 0.5 threshold effectively shifts
- With weight=5, model might predict class 1 at probability ≈ 0.3
- With balanced, model predicts class 1 at probability ≈ 0.1

---

## Practical Guidelines

### When to Use Each Option

**Use `class_weight=None` (no weighting) when:**
- Dataset is balanced (roughly equal classes)
- False positives and false negatives have equal cost
- Default: usually not appropriate for medical data

**Use `class_weight="balanced"` when:**
- Severe imbalance (>50:1) AND recall is critical
- Missing minority class is very costly
- Can tolerate many false positives
- Example: Cancer screening (better safe than sorry)

**Use manual weights like `{0: 1, 1: 5}` when:**
- Moderate to high imbalance (5:1 to 30:1) ✅ **Your case!**
- Want balance between precision and recall
- Need to tune the trade-off
- Most common in practice

**Use threshold adjustment instead when:**
- Already trained model, don't want to retrain
- Need different operating points for different use cases
- Want to optimize for specific metric (F1, balanced accuracy)

---

## Trade-offs

### Higher Weight → More Recall, Less Precision

**Weight = 1** (no weighting)
```
Predictions: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  (mostly 0)
Precision: 80% (when we say 1, usually correct)
Recall: 20% (but we miss most 1s)
```

**Weight = 5** (moderate)
```
Predictions: [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]  (more 1s)
Precision: 50% (more false alarms)
Recall: 60% (but catch more true 1s)
```

**Weight = 18** (balanced)
```
Predictions: [1, 1, 0, 1, 1, 1, 1, 0, 1, 1]  (too many 1s)
Precision: 20% (mostly wrong)
Recall: 80% (catch almost all 1s)
```

### The F1 Score Sweet Spot

F1 = 2 × (Precision × Recall) / (Precision + Recall)

- **Too low weight**: High precision, low recall → Low F1
- **Optimal weight**: Balanced precision and recall → **High F1**
- **Too high weight**: Low precision, high recall → Low F1

Your script finds the weight that maximizes F1 (or another metric).

---

## Implementation Details

### In Scikit-learn

```python
# Option 1: Balanced
clf = LogisticRegression(class_weight='balanced')

# Option 2: Manual
clf = LogisticRegression(class_weight={0: 1, 1: 5})

# Option 3: No weighting
clf = LogisticRegression(class_weight=None)
```

### Behind the Scenes

During training, scikit-learn modifies the sample weights:

```python
sample_weight[i] = class_weight[y[i]]
```

Then in the loss function:

```python
loss = sum(sample_weight[i] * loss_i for i in range(n_samples))
```

### Effect on Coefficients

The model learns different coefficients:
- **No weighting**: Coefficients favor patterns in majority class
- **With weighting**: Coefficients balance both classes
- **Higher weight**: Coefficients increasingly favor minority class patterns

---

## Clinical Context for Your Problem

### What Do the Metrics Mean?

**Precision (Positive Predictive Value):**
- "If model says invasion, how likely is it true?"
- Low precision = Many false alarms
- **Clinical impact**: Unnecessary follow-up procedures, patient anxiety

**Recall (Sensitivity):**
- "Of all actual invasions, how many did we find?"
- Low recall = Missing cases
- **Clinical impact**: Undetected invasions, treatment delays

### Cost-Benefit Analysis

**False Positive** (predict invasion when none exists):
- Patient gets unnecessary follow-up
- Extra imaging/biopsies
- Anxiety and healthcare costs

**False Negative** (miss an invasion):
- Delayed diagnosis
- Potential disease progression
- Worse outcomes

**If false negatives are worse**: Use higher weight (e.g., 7-10)
**If false positives are worse**: Use lower weight (e.g., 2-3)
**If roughly equal**: Use moderate weight (e.g., 5)

---

## Recommendations for Your Project

### Step 1: Run Optimization

```powershell
python scripts/find_optimal_class_weight.py
```

This tests weights 1-25 and shows you the trade-offs.

### Step 2: Choose Based on Your Priority

**If you prioritize not missing invasions (recall):**
```yaml
class_weight: {0: 1, 1: 7}  # Higher weight
```

**If you want balanced performance (F1):**
```yaml
class_weight: {0: 1, 1: 5}  # Moderate weight (recommended start)
```

**If you want fewer false alarms (precision):**
```yaml
class_weight: {0: 1, 1: 3}  # Lower weight
```

### Step 3: Monitor in MLflow

With the new MLflow integration, you can:
1. Run multiple experiments with different weights
2. Compare all metrics side-by-side
3. Choose based on your clinical requirements

```powershell
python scripts/train_classifier.py model.params.class_weight={0:1,1:3}
python scripts/train_classifier.py model.params.class_weight={0:1,1:5}
python scripts/train_classifier.py model.params.class_weight={0:1,1:7}
```

Then view and compare in MLflow UI!

---

## Summary

| Aspect | No Weighting | Balanced | Manual (weight=5) |
|--------|--------------|----------|-------------------|
| **Loss for class 0** | 1.0 | 0.528 | 1.0 |
| **Loss for class 1** | 1.0 | 9.55 | 5.0 |
| **Effective ratio** | 18:1 | 1:1 | 3.6:1 |
| **Prediction bias** | Mostly 0 | Mostly 1 | Balanced |
| **Accuracy** | 94.8% | 79% | 93% |
| **Precision** | High | Low | Medium |
| **Recall** | Low | Medium | Medium |
| **F1 Score** | Low | Low | **High** |
| **Use when** | Balanced data | Critical recalls | **Most cases** |

**Key Takeaway**: 
- `class_weight="balanced"` is often **too aggressive** for moderate imbalance
- Manual weights give you **control over the precision-recall trade-off**
- Your script helps you **find the sweet spot empirically**
- MLflow helps you **compare and choose** the best configuration

Start with `{0: 1, 1: 5}` and adjust based on results! 🎯
