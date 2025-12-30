# =============================================================================
# Random Forest Classifier with GridSearchCV for Faller Prediction
# Optimized for ROC-AUC with Cross-Validation and Overfitting Prevention
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, accuracy_score, f1_score
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("sway_static_featuers_minimized.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nTarget distribution:\n{df['faller'].value_counts()}")

# Define meta columns (non-features) - exclude from training
# Note: age_num, sex, height, weight, BMI ARE included as features
meta_cols = ["part_id", "group", "recorded_in_the_lab", "faller"]

# Feature columns (all numerical sway features)
feature_cols = [c for c in df.columns if c not in meta_cols]
print(f"\nNumber of features: {len(feature_cols)}")

# Prepare X and y
X = df[feature_cols]
y = df["faller"]

# Train-Test Split (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train class distribution:\n{y_train.value_counts()}")

# =============================================================================
# Pipeline with Imputation, Scaling, and Random Forest
# =============================================================================
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler()),  # Normalize features
    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# =============================================================================
# GridSearchCV Parameter Grid
# Conservative parameters to prevent overfitting
# =============================================================================
param_grid = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [3, 5, 7, 10, None],  # Limiting depth prevents overfitting
    'rf__min_samples_split': [5, 10, 15],  # Higher values prevent overfitting
    'rf__min_samples_leaf': [3, 5, 7],  # Higher values prevent overfitting
    'rf__max_features': ['sqrt', 'log2', 0.3],  # Limit features per tree
}

# =============================================================================
# Stratified K-Fold Cross Validation
# =============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =============================================================================
# GridSearchCV with ROC-AUC scoring
# =============================================================================
print("\n" + "="*60)
print("Starting GridSearchCV...")
print("="*60)

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1,
    return_train_score=True  # To detect overfitting
)

grid_search.fit(X_train, y_train)

# =============================================================================
# Results
# =============================================================================
print("\n" + "="*60)
print("GRIDSEACHCV RESULTS")
print("="*60)
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC Score: {grid_search.best_score_:.4f}")

# Check for overfitting (compare train vs CV scores)
best_idx = grid_search.best_index_
train_score = grid_search.cv_results_['mean_train_score'][best_idx]
cv_score = grid_search.cv_results_['mean_test_score'][best_idx]
print(f"\nTrain ROC-AUC: {train_score:.4f}")
print(f"CV ROC-AUC: {cv_score:.4f}")
print(f"Overfitting Gap: {train_score - cv_score:.4f}")

if train_score - cv_score > 0.15:
    print("⚠️  Warning: Possible overfitting detected!")
else:
    print("✓ Model appears well-generalized")

# =============================================================================
# Evaluate on Test Set
# =============================================================================
print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_roc_auc = roc_auc_score(y_test, y_pred_proba)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print(f"\nTest ROC-AUC: {test_roc_auc:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Faller', 'Faller']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# =============================================================================
# Additional Cross-Validation on Full Training Data
# =============================================================================
print("\n" + "="*60)
print("CROSS-VALIDATION SCORES (Best Model on Train Set)")
print("="*60)

cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f"CV Scores: {cv_scores}")
print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =============================================================================
# Feature Importance (Top 15)
# =============================================================================
print("\n" + "="*60)
print("TOP 15 FEATURE IMPORTANCES")
print("="*60)

rf_model = best_model.named_steps['rf']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(15).to_string(index=False))

# =============================================================================
# Visualization
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {test_roc_auc:.4f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend(loc="lower right")

# 2. Feature Importance (Top 10)
top_features = feature_importance.head(10)
axes[1].barh(range(len(top_features)), top_features['importance'].values)
axes[1].set_yticks(range(len(top_features)))
axes[1].set_yticklabels(top_features['feature'].values)
axes[1].invert_yaxis()
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 10 Feature Importances')

# 3. CV Score Distribution
axes[2].bar(range(1, 6), cv_scores, color='steelblue', alpha=0.7)
axes[2].axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
axes[2].set_xlabel('Fold')
axes[2].set_ylabel('ROC-AUC')
axes[2].set_title('Cross-Validation Scores')
axes[2].legend()
axes[2].set_ylim([0, 1])

plt.tight_layout()
plt.savefig('rf_classifier_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Best Model: RandomForestClassifier")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"CV ROC-AUC: {grid_search.best_score_:.4f}")
print(f"Test ROC-AUC: {test_roc_auc:.4f}")
print(f"\nResults saved to 'rf_classifier_results.png'")
