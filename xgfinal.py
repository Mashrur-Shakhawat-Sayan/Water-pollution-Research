import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, balanced_accuracy_score, classification_report
)
from sklearn.utils import class_weight
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ------------------ 1. Load Dataset ------------------
print("📂 Loading dataset...")
df = pd.read_csv("final_cleaned_data.csv")
print(f"✅ Loaded! Shape: {df.shape}")

# ------------------ 2. Features & Targets ------------------
features = [
    'Ammonia_(mg/l)', 'Biochemical_Oxygen_Demand_(mg/l)',
    'Dissolved_Oxygen_(mg/l)', 'Orthophosphate_(mg/l)',
    'pH', 'Temperature_(°C)', 'Nitrogen_(mg/l)', 'Nitrate_(mg/l)'
]

# Clean data first
print("🧹 Cleaning data...")
df_clean = df[features + ['CCME_Values', 'CCME_WQI']].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna()
print(f"✅ Cleaned! New shape: {df_clean.shape}")

X = df_clean[features]
y_regression = df_clean['CCME_Values']

label_mapping = {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Marginal': 3, 'Poor': 4}
y_classification = df_clean['CCME_WQI'].map(label_mapping)

# ------------------ 3. Temporal Split ------------------
train_idx = int(0.64 * len(df_clean))
val_idx = int(0.8 * len(df_clean))

X_train, y_train_reg, y_train_class = X.iloc[:train_idx], y_regression.iloc[:train_idx], y_classification.iloc[:train_idx]
X_val,   y_val_reg, y_val_class     = X.iloc[train_idx:val_idx], y_regression.iloc[train_idx:val_idx], y_classification.iloc[train_idx:val_idx]
X_test,  y_test_reg, y_test_class   = X.iloc[val_idx:], y_regression.iloc[val_idx:], y_classification.iloc[val_idx:]

print(f"⏳ Time split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ------------------ 4. Correlation Analysis (on training data only) ------------------
train_data = pd.concat([X_train, y_train_reg], axis=1)
correlations = train_data.corr()['CCME_Values'].drop('CCME_Values')
print("\n🔍 Feature correlations with CCME_Values (training data only):")
print(correlations.sort_values(ascending=False))

# Lenient: drop top-2 correlated
top2_corr = correlations.abs().sort_values(ascending=False).head(2).index.tolist()
lenient_features = [f for f in features if f not in top2_corr]

# Strict: drop all with |corr| > 0.85
strict_features = [f for f, c in correlations.items() if abs(c) <= 0.85]

print(f"\n⚠️ Lenient drop: {top2_corr}")
print(f"⚠️ Strict drop: {[f for f in features if f not in strict_features]}")

# ------------------ 5. Helper Functions ------------------
def run_regression(X_train, y_train, X_val, y_val, X_test, y_test, gpu_params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        **gpu_params
    }

    model = xgb.train(params, dtrain,
                      num_boost_round=500,
                      evals=[(dtrain, 'train'), (dval, 'val')],
                      early_stopping_rounds=30,
                      verbose_eval=False)

    y_pred = model.predict(dtest)
    
    # Also predict on validation set to check for overfitting
    y_val_pred = model.predict(dval)
    val_r2 = r2_score(y_val, y_val_pred)
    
    return model, y_pred, val_r2

def run_classification(X_train, y_train, X_val, y_val, X_test, y_test, sample_weights, gpu_params):
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'multi:softprob',
        'num_class': 5,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        **gpu_params
    }

    model = xgb.train(params, dtrain,
                      num_boost_round=500,
                      evals=[(dtrain, 'train'), (dval, 'val')],
                      early_stopping_rounds=30,
                      verbose_eval=False)

    y_pred_proba = model.predict(dtest)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Also predict on validation set to check for overfitting
    y_val_pred_proba = model.predict(dval)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    return model, y_pred, val_acc

# ------------------ 6. Run Both Versions ------------------
gpu_params = {'tree_method': 'gpu_hist', 'device': 'cuda', 'predictor': 'gpu_predictor'}

results = {}

for version, fset in [("Lenient", lenient_features), ("Strict", strict_features)]:
    print("\n" + "="*30)
    print(f"🚀 Training with {version} Feature Selection")
    print("="*30)
    print(f"Using features: {fset}")

    # Prepare splits
    X_train_fs, X_val_fs, X_test_fs = X_train[fset], X_val[fset], X_test[fset]

    # Calculate class weights for this feature set
    classes = np.unique(y_train_class)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train_class)
    class_weights = dict(zip(classes, weights))
    sample_weights = np.array([class_weights[label] for label in y_train_class])

    # Regression
    reg_model, y_pred_reg, val_r2_reg = run_regression(X_train_fs, y_train_reg, X_val_fs, y_val_reg, X_test_fs, y_test_reg, gpu_params)
    r2 = r2_score(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    # Classification
    cls_model, y_pred_class, val_acc_cls = run_classification(X_train_fs, y_train_class, X_val_fs, y_val_class,
                                                 X_test_fs, y_test_class, sample_weights, gpu_params)
    acc = accuracy_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class, average='macro')
    bal_acc = balanced_accuracy_score(y_test_class, y_pred_class)

    print(f"\n📊 Regression ({version}) → R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    print(f"   Validation R²: {val_r2_reg:.4f} (Overfitting gap: {r2 - val_r2_reg:.4f})")
    
    print(f"📊 Classification ({version}) → Acc={acc:.4f}, F1={f1:.4f}, BalAcc={bal_acc:.4f}")
    print(f"   Validation Acc: {val_acc_cls:.4f} (Overfitting gap: {acc - val_acc_cls:.4f})")
    
    if (r2 - val_r2_reg > 0.1) or (acc - val_acc_cls > 0.1):
        print("⚠️  Significant overfitting detected!")
    
    print("\nClassification Report:")
    print(classification_report(y_test_class, y_pred_class, target_names=label_mapping.keys()))

    results[version] = {
        "R2": r2, "MAE": mae, "RMSE": rmse, 
        "Acc": acc, "F1": f1, "BalAcc": bal_acc,
        "Val_R2": val_r2_reg, "Val_Acc": val_acc_cls
    }

# ------------------ 7. Plot Comparison ------------------
metrics = ["R2", "MAE", "RMSE", "Acc", "F1", "BalAcc"]

plt.figure(figsize=(12, 8))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    vals = [results["Lenient"][metric], results["Strict"][metric]]
    plt.bar(["Lenient", "Strict"], vals, color=['skyblue', 'salmon'])
    plt.title(metric)
    plt.ylabel(metric)
    
    # Add value labels on top of bars
    for j, v in enumerate(vals):
        plt.text(j, v + 0.01 * max(vals), f"{v:.4f}", ha='center')

plt.tight_layout()
plt.savefig("XGBoost-lenient_vs_strict_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ------------------ 8. Final Analysis ------------------
print("\n" + "="*50)
print("📋 FINAL ANALYSIS")
print("="*50)

for version in ["Lenient", "Strict"]:
    r2_gap = results[version]["R2"] - results[version]["Val_R2"]
    acc_gap = results[version]["Acc"] - results[version]["Val_Acc"]
    
    print(f"\n{version} Model:")
    print(f"  Regression overfitting gap: {r2_gap:.4f} {'(⚠️ High)' if r2_gap > 0.1 else '(✅ Good)'}")
    print(f"  Classification overfitting gap: {acc_gap:.4f} {'(⚠️ High)' if acc_gap > 0.1 else '(✅ Good)'}")

# Check if strict features performed similarly to lenient
r2_diff = results["Lenient"]["R2"] - results["Strict"]["R2"]
acc_diff = results["Lenient"]["Acc"] - results["Strict"]["Acc"]

print(f"\nPerformance difference (Lenient - Strict):")
print(f"  R²: {r2_diff:.4f} {'(⚠️ Big drop)' if r2_diff > 0.1 else '(✅ Minimal)'}")
print(f"  Accuracy: {acc_diff:.4f} {'(⚠️ Big drop)' if acc_diff > 0.1 else '(✅ Minimal)'}")

if abs(r2_diff) < 0.05 and abs(acc_diff) < 0.05:
    print("\n🎉 Strict feature selection performed similarly to lenient!")
    print("   This suggests the removed features were indeed causing data leakage.")
else:
    print("\n🤔 Strict feature selection performed worse than lenient.")
    print("   The removed features might contain genuine predictive signal.")