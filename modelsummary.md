# Random Forest Faller Prediction Model

## Dataset
- **File**: `sway_static_featuers_minimized.csv`
- **Samples**: 90 (66 non-fallers, 24 fallers)
- **Target**: `faller` (binary classification)

## Features Used (21 total)

| Category | Features |
|----------|----------|
| **Demographics** | `age_num`, `sex`, `height`, `weight`, `BMI` |
| **Aggregate Sway** | `AREA_mean`, `MDIST_mean`, `MFREQ_mean`, `MVELO_mean`, `RDIST_mean`, `TOTEX_mean` |
| **AP Direction** | `MDIST_AP_mean`, `MFREQ_AP_mean`, `MVELO_AP_mean`, `RDIST_AP_mean`, `TOTEX_AP_mean` |
| **ML Direction** | `MDIST_ML_mean`, `MFREQ_ML_mean`, `MVELO_ML_mean`, `RDIST_ML_mean`, `TOTEX_ML_mean` |

## Model Configuration

| Parameter | Value |
|-----------|-------|
| **Algorithm** | Random Forest Classifier |
| **n_estimators** | 300 |
| **max_depth** | 3 |
| **min_samples_leaf** | 3 |
| **min_samples_split** | 5 |
| **max_features** | 0.3 |
| **class_weight** | balanced |

## Performance

| Metric | Score |
|--------|-------|
| **Test ROC-AUC** | **0.8615** |
| CV ROC-AUC | 0.6198 |
| Test Accuracy | 0.7222 |

## Top 5 Important Features

1. `age_num` (20.0%)
2. `BMI` (16.0%)
3. `RDIST_AP_mean` (7.7%)
4. `MFREQ_ML_mean` (6.6%)
5. `MDIST_mean` (4.7%)

## Code Files

- **Main model**: [rf_classifier_code.py](file:///home/timo/mdf/rf_classifier_code.py)
- **Results plot**: `rf_classifier_results.png`
