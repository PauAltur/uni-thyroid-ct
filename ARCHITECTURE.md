# Modular Classification Architecture

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                  kfold_classification.py (Main Script)              │
│                                                                     │
│  run_classification_pipeline()                                     │
│  ├─ Parse arguments                                                │
│  ├─ Call pipeline function                                         │
│  └─ Handle errors                                                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              run_classification_pipeline() Function                 │
│                                                                     │
│  Orchestrates all modules in sequence:                             │
│  1. Load data        → data_loader.py                              │
│  2. Validate data    → data_loader.py                              │
│  3. Create classifier → models.py                                  │
│  4. Run CV           → cross_validation.py                         │
│  5. Compute metrics  → metrics.py                                  │
│  6. Aggregate metrics → metrics.py                                 │
│  7. Save results     → results_io.py                               │
│  8. Visualize        → visualization.py                            │
└─────────────────────────────────────────────────────────────────────┘
           │              │              │              │
           ▼              ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ data_loader  │ │   models     │ │cross_valid.  │ │   metrics    │
│              │ │              │ │              │ │              │
│ - Load H5    │ │ - Get clf    │ │ - GroupKFold │ │ - Compute    │
│ - Filter     │ │ - Logistic   │ │ - Train fold │ │ - Aggregate  │
│ - Extract    │ │ - RandomForest│ │ - Evaluate  │ │ - Log        │
│ - Validate   │ │ - GradBoost  │ │ - No leakage │ │              │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
                                                            │
           ┌────────────────────────────────────────────────┴────────┐
           ▼                                                         ▼
┌──────────────────┐                                    ┌──────────────────┐
│  visualization   │                                    │   results_io     │
│                  │                                    │                  │
│ - Plot CV results│                                    │ - Save CSV       │
│ - ROC curves     │                                    │ - Save JSON      │
│ - Confusion matrix│                                   │ - Save Excel     │
│ - Box plots      │                                    │ - Load results   │
└──────────────────┘                                    └──────────────────┘
```

## Data Flow

```
┌─────────┐
│ H5 File │
└────┬────┘
     │
     ▼
┌──────────────────┐
│  data_loader.py  │  Load embeddings + metadata
│                  │  Filter by tissue threshold
└────┬─────────────┘
     │
     ├─► embeddings (X)
     ├─► labels (y)
     └─► sample_codes (groups)
         │
         ▼
    ┌────────────────┐
    │   models.py    │  Create classifier instance
    └────┬───────────┘
         │
         ├─► clf (sklearn classifier)
         │
         ▼
    ┌─────────────────────┐
    │cross_validation.py  │  K-fold split by sample
    │                     │  Train & evaluate each fold
    └────┬────────────────┘
         │
         ├─► fold_metadata
         ├─► y_true_list
         └─► y_pred_list
             │
             ▼
        ┌─────────────┐
        │ metrics.py  │  Compute metrics per fold
        │             │  Aggregate across folds
        └────┬────────┘
             │
             ├─► fold_results
             └─► aggregated_metrics
                 │
                 ├──────────────┬──────────────┐
                 ▼              ▼              ▼
          ┌──────────┐   ┌──────────┐  ┌──────────┐
          │results_io│   │visualiz. │  │ Console  │
          │          │   │          │  │          │
          │ CSV/JSON │   │ PNG plots│  │ Logging  │
          └──────────┘   └──────────┘  └──────────┘
```

## Module Responsibilities

### data_loader.py
- **Input**: HDF5 file path, tissue threshold
- **Output**: Embeddings array, metadata DataFrame
- **Does**: Loading, filtering, validation

### models.py
- **Input**: Classifier type, hyperparameters
- **Output**: Scikit-learn classifier instance
- **Does**: Factory pattern for classifiers

### cross_validation.py
- **Input**: Data (X, y, groups), classifier, n_folds
- **Output**: Predictions and metadata per fold
- **Does**: GroupKFold splitting, training, prediction

### metrics.py
- **Input**: True labels, predictions, probabilities
- **Output**: Metrics dictionaries
- **Does**: Compute and aggregate metrics

### visualization.py
- **Input**: Results dictionaries
- **Output**: PNG plot files
- **Does**: Create comprehensive visualizations

### results_io.py
- **Input**: Results dictionaries, configuration
- **Output**: CSV, JSON, Excel files
- **Does**: Save and load results

## Key Design Principles

1. **Single Responsibility**: Each module does one thing well
2. **Loose Coupling**: Modules interact through well-defined interfaces
3. **High Cohesion**: Related functions grouped together
4. **Testability**: Each module can be tested independently
5. **Reusability**: Modules can be used in other projects
6. **Extensibility**: Easy to add new features

## Example: Adding a New Classifier

```python
# 1. Edit models.py
DEFAULT_PARAMS["xgboost"] = {
    "n_estimators": 100,
    "max_depth": 6,
}

def get_classifier(classifier_type, random_state, **kwargs):
    # ... existing code ...
    elif classifier_type == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(**params)
    return clf

# 2. Update command-line choices in main script
parser.add_argument(
    "--classifier",
    choices=["logistic", "random_forest", "gradient_boosting", "svm", "xgboost"],
)

# 3. Done! No changes needed to other modules
```

## Comparison: Before vs After

### Before (Monolithic)
```
PatchClassifier (636 lines)
├── __init__()
├── load_data()
├── filter_by_tissue()
├── get_classifier()
├── compute_metrics()
├── run_cross_validation()
├── compute_aggregated_metrics()
├── save_results()
├── plot_results()
└── run()
```

**Issues**:
- Hard to test individual components
- Difficult to reuse specific functionality
- Changes in one area affect entire class
- Not modular or extensible

### After (Modular)
```
run_classification_pipeline() (170 lines)
│
├── data_loader.py (175 lines)
│   ├── load_embeddings_from_h5()
│   ├── filter_by_tissue_percentage()
│   ├── get_sample_groups()
│   ├── get_labels()
│   └── validate_data_for_cv()
│
├── models.py (115 lines)
│   ├── get_classifier()
│   └── get_available_classifiers()
│
├── cross_validation.py (220 lines)
│   ├── get_group_kfold_splits()
│   ├── train_and_evaluate_fold()
│   └── run_cross_validation()
│
├── metrics.py (145 lines)
│   ├── compute_classification_metrics()
│   ├── aggregate_fold_metrics()
│   ├── log_metrics()
│   └── log_aggregated_metrics()
│
├── visualization.py (260 lines)
│   ├── plot_cv_results()
│   ├── plot_roc_curves()
│   └── create_results_summary_table()
│
└── results_io.py (180 lines)
    ├── save_results()
    ├── load_results()
    ├── create_filename_base()
    └── export_to_excel()
```

**Benefits**:
- ✅ Each module is independently testable
- ✅ Easy to reuse specific functionality
- ✅ Changes are isolated to relevant modules
- ✅ Highly modular and extensible
- ✅ Better documentation and organization
