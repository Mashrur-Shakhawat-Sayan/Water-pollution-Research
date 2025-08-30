import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import model-specific libraries
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, balanced_accuracy_score, 
    classification_report, precision_score, recall_score, log_loss
)
from sklearn.utils import class_weight
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from tqdm import tqdm

# ------------------ 1. Load and Prepare Data ------------------
print("📂 Loading dataset...")
df = pd.read_csv("final_cleaned_data.csv")
print(f"✅ Loaded! Shape: {df.shape}")

# Find date column for temporal sorting
date_col = None
for col in df.columns:
    if 'date' in col.lower():
        date_col = col
        break

if date_col:
    print(f"📅 Found date column: {date_col}")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.sort_values(date_col).reset_index(drop=True)
    print("✅ Data sorted by date for proper temporal splitting")
else:
    print("⚠️ No date column found - using existing order")

# Features
features = [
    'Ammonia_(mg/l)', 'Biochemical_Oxygen_Demand_(mg/l)',
    'Dissolved_Oxygen_(mg/l)', 'Orthophosphate_(mg/l)',
    'pH', 'Temperature_(°C)', 'Nitrogen_(mg/l)', 'Nitrate_(mg/l)'
]

# Clean data
print("🧹 Cleaning data...")
df_clean = df[features + ['CCME_Values', 'CCME_WQI']].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
df_clean = df_clean.dropna()
print(f"✅ Cleaned! New shape: {df_clean.shape}")

X = df_clean[features]
y_regression = df_clean['CCME_Values']

label_mapping = {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Marginal': 3, 'Poor': 4}
y_classification = df_clean['CCME_WQI'].map(label_mapping)

# Temporal split (64% train, 16% validation, 20% test)
train_idx = int(0.64 * len(df_clean))
val_idx = int(0.8 * len(df_clean))

X_train, y_train_reg, y_train_class = X.iloc[:train_idx], y_regression.iloc[:train_idx], y_classification.iloc[:train_idx]
X_val, y_val_reg, y_val_class = X.iloc[train_idx:val_idx], y_regression.iloc[train_idx:val_idx], y_classification.iloc[train_idx:val_idx]
X_test, y_test_reg, y_test_class = X.iloc[val_idx:], y_regression.iloc[val_idx:], y_classification.iloc[val_idx:]

print(f"⏳ Time split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ------------------ 2. Feature Correlation Analysis ------------------
print("\n🔍 Analyzing feature correlations...")

# For regression
train_data_reg = pd.concat([X_train, y_train_reg], axis=1)
correlations_reg = train_data_reg.corr()['CCME_Values'].drop('CCME_Values')
print("\nRegression target (CCME_Values) correlations:")
print(correlations_reg.sort_values(ascending=False))

# For classification
train_data_cls = pd.concat([X_train, y_train_class.rename('CCME_WQI_encoded')], axis=1)
correlations_cls = train_data_cls.corr()['CCME_WQI_encoded'].drop('CCME_WQI_encoded')
print("\nClassification target (CCME_WQI encoded) correlations:")
print(correlations_cls.sort_values(ascending=False))

# Define feature sets
top3_corr_reg = correlations_reg.abs().sort_values(ascending=False).head(3).index.tolist()
top3_corr_cls = correlations_cls.abs().sort_values(ascending=False).head(3).index.tolist()

print(f"\n⚠️ Top 3 correlated features for regression: {top3_corr_reg}")
print(f"⚠️ Top 3 correlated features for classification: {top3_corr_cls}")

# Create feature sets
feature_sets = {
    'All_Features': features.copy(),
    'No_Top3_Reg': [f for f in features if f not in top3_corr_reg],
    'No_Top3_Cls': [f for f in features if f not in top3_corr_cls]
}

# ------------------ 3. Model Evaluation Functions ------------------
def evaluate_xgboost(feature_set_name, feature_set):
    print(f"\n" + "="*50)
    print(f"🚀 Training XGBoost - {feature_set_name}")
    print("="*50)
    
    results = {}
    start_time = time.time()
    
    # Select features
    X_train_fs = X_train[feature_set]
    X_val_fs = X_val[feature_set]
    X_test_fs = X_test[feature_set]
    
    # GPU parameters
    gpu_params = {'tree_method': 'gpu_hist', 'device': 'cuda', 'predictor': 'gpu_predictor'}
    
    # Regression with progress bar
    print("📊 Training regression model...")
    dtrain_reg = xgb.DMatrix(X_train_fs, label=y_train_reg)
    dval_reg = xgb.DMatrix(X_val_fs, label=y_val_reg)
    dtest_reg = xgb.DMatrix(X_test_fs, label=y_test_reg)
    
    params_reg = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        **gpu_params
    }
    
    eval_results_reg = {}
    model_reg = xgb.train(params_reg, dtrain_reg,
                          num_boost_round=500,
                          evals=[(dtrain_reg, 'train'), (dval_reg, 'val')],
                          early_stopping_rounds=30,
                          verbose_eval=False,
                          evals_result=eval_results_reg)
    
    y_pred_reg = model_reg.predict(dtest_reg)
    
    # Classification with progress bar
    print("📊 Training classification model...")
    classes = np.unique(y_train_class)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train_class)
    class_weights = dict(zip(classes, weights))
    sample_weights = np.array([class_weights[label] for label in y_train_class])
    
    dtrain_cls = xgb.DMatrix(X_train_fs, label=y_train_class, weight=sample_weights)
    dval_cls = xgb.DMatrix(X_val_fs, label=y_val_class)
    dtest_cls = xgb.DMatrix(X_test_fs, label=y_test_class)
    
    params_cls = {
        'objective': 'multi:softprob',
        'num_class': 5,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'eval_metric': ['mlogloss', 'merror'],
        **gpu_params
    }
    
    eval_results_cls = {}
    model_cls = xgb.train(params_cls, dtrain_cls,
                          num_boost_round=500,
                          evals=[(dtrain_cls, 'train'), (dval_cls, 'val')],
                          early_stopping_rounds=30,
                          verbose_eval=False,
                          evals_result=eval_results_cls)
    
    y_pred_cls_proba = model_cls.predict(dtest_cls)
    y_pred_cls = np.argmax(y_pred_cls_proba, axis=1)
    
    # Calculate metrics
    results['Regression'] = {
        'R2': r2_score(y_test_reg, y_pred_reg),
        'MAE': mean_absolute_error(y_test_reg, y_pred_reg),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    }
    
    results['Classification'] = {
        'Accuracy': accuracy_score(y_test_class, y_pred_cls),
        'F1': f1_score(y_test_class, y_pred_cls, average='macro'),
        'Balanced_Accuracy': balanced_accuracy_score(y_test_class, y_pred_cls),
        'LogLoss': log_loss(y_test_class, y_pred_cls_proba)
    }
    
    results['Training_Time'] = time.time() - start_time
    results['Feature_Set'] = feature_set_name
    results['Features'] = feature_set
    
    return results

def evaluate_svm():
    print("\n" + "="*50)
    print("🚀 Training SVM")
    print("="*50)
    
    results = {}
    start_time = time.time()
    
    # Regression with progress bar
    print("📊 Training regression model...")
    svr_model = LinearSVR(random_state=42, max_iter=10000)
    with tqdm(total=100, desc="SVR Training") as pbar:
        svr_model.fit(X_train, y_train_reg)
        pbar.update(100)
    
    y_pred_reg = svr_model.predict(X_test)
    
    # Classification with progress bar
    print("📊 Training classification model...")
    svc_model = LinearSVC(class_weight='balanced', max_iter=5000, random_state=42)
    with tqdm(total=100, desc="SVC Training") as pbar:
        svc_model.fit(X_train, y_train_class)
        pbar.update(100)
    
    y_pred_cls = svc_model.predict(X_test)
    
    # Calculate metrics
    results['Regression'] = {
        'R2': r2_score(y_test_reg, y_pred_reg),
        'MAE': mean_absolute_error(y_test_reg, y_pred_reg),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    }
    
    results['Classification'] = {
        'Accuracy': accuracy_score(y_test_class, y_pred_cls),
        'F1': f1_score(y_test_class, y_pred_cls, average='macro'),
        'Balanced_Accuracy': balanced_accuracy_score(y_test_class, y_pred_cls),
        'LogLoss': 'N/A'  # LinearSVC doesn't provide probabilities
    }
    
    results['Training_Time'] = time.time() - start_time
    results['Feature_Set'] = 'All_Features'
    results['Features'] = features
    
    return results

def evaluate_random_forest(feature_set_name, feature_set):
    print(f"\n" + "="*50)
    print(f"🚀 Training Random Forest - {feature_set_name}")
    print("="*50)
    
    results = {}
    start_time = time.time()
    
    # Select features
    X_train_fs = X_train[feature_set]
    X_val_fs = X_val[feature_set]
    X_test_fs = X_test[feature_set]
    
    # Combine train and validation sets for CV
    X_train_val = pd.concat([X_train_fs, X_val_fs])
    y_train_val_reg = pd.concat([y_train_reg, y_val_reg])
    y_train_val_class = pd.concat([y_train_class, y_val_class])
    
    # Reduce dataset size for Random Forest to avoid memory issues
    # Use a subset of the data for training
    sample_size = min(100000, len(X_train_val))  # Limit to 100,000 samples
    if len(X_train_val) > sample_size:
        indices = np.random.choice(len(X_train_val), sample_size, replace=False)
        X_train_val = X_train_val.iloc[indices]
        y_train_val_reg = y_train_val_reg.iloc[indices]
        y_train_val_class = y_train_val_class.iloc[indices]
    
    # Set up cross-validation with fewer splits
    n_splits = 3  # Reduce from 5 to 3 to save memory
    kf = KFold(n_splits=n_splits, shuffle=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    
    # Use smaller Random Forest parameters to reduce memory usage
    rf_params = {
        'n_estimators': 50,  # Reduce from 100 to 50
        'max_depth': 10,      # Limit tree depth
        'min_samples_split': 20,  # Increase to reduce tree size
        'min_samples_leaf': 10,   # Increase to reduce tree size
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Regression CV with progress bar
    print("📊 Running regression cross-validation...")
    r2_scores, mae_scores, rmse_scores = [], [], []
    
    for train_index, val_index in tqdm(kf.split(X_train_val), total=n_splits, desc="Regression CV"):
        X_train_cv, X_val_cv = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train_cv, y_val_cv = y_train_val_reg.iloc[train_index], y_train_val_reg.iloc[val_index]
        
        rf_reg = RandomForestRegressor(**rf_params)
        rf_reg.fit(X_train_cv, y_train_cv)
        y_pred = rf_reg.predict(X_val_cv)
        
        r2_scores.append(r2_score(y_val_cv, y_pred))
        mae_scores.append(mean_absolute_error(y_val_cv, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_pred)))
    
    # Train final regression model on all training+validation data
    rf_reg_final = RandomForestRegressor(**rf_params)
    rf_reg_final.fit(X_train_val, y_train_val_reg)
    y_pred_reg = rf_reg_final.predict(X_test_fs)
    
    # Classification CV with progress bar
    print("📊 Running classification cross-validation...")
    acc_scores, f1_scores, bal_acc_scores, logloss_scores = [], [], [], []
    
    for train_index, val_index in tqdm(skf.split(X_train_val, y_train_val_class), total=n_splits, desc="Classification CV"):
        X_train_cv, X_val_cv = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train_cv, y_val_cv = y_train_val_class.iloc[train_index], y_train_val_class.iloc[val_index]
        
        rf_cls = RandomForestClassifier(**rf_params)
        rf_cls.fit(X_train_cv, y_train_cv)
        y_pred = rf_cls.predict(X_val_cv)
        y_pred_proba = rf_cls.predict_proba(X_val_cv)
        
        acc_scores.append(accuracy_score(y_val_cv, y_pred))
        f1_scores.append(f1_score(y_val_cv, y_pred, average='macro'))
        bal_acc_scores.append(balanced_accuracy_score(y_val_cv, y_pred))
        logloss_scores.append(log_loss(y_val_cv, y_pred_proba))
    
    # Train final classification model on all training+validation data
    rf_cls_final = RandomForestClassifier(**rf_params)
    rf_cls_final.fit(X_train_val, y_train_val_class)
    y_pred_cls = rf_cls_final.predict(X_test_fs)
    y_pred_cls_proba = rf_cls_final.predict_proba(X_test_fs)
    
    # Calculate metrics
    results['Regression'] = {
        'R2': r2_score(y_test_reg, y_pred_reg),
        'MAE': mean_absolute_error(y_test_reg, y_pred_reg),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
        'CV_R2_mean': np.mean(r2_scores),
        'CV_R2_std': np.std(r2_scores),
        'CV_MAE_mean': np.mean(mae_scores),
        'CV_MAE_std': np.std(mae_scores),
        'CV_RMSE_mean': np.mean(rmse_scores),
        'CV_RMSE_std': np.std(rmse_scores)
    }
    
    results['Classification'] = {
        'Accuracy': accuracy_score(y_test_class, y_pred_cls),
        'F1': f1_score(y_test_class, y_pred_cls, average='macro'),
        'Balanced_Accuracy': balanced_accuracy_score(y_test_class, y_pred_cls),
        'LogLoss': log_loss(y_test_class, y_pred_cls_proba),
        'CV_Accuracy_mean': np.mean(acc_scores),
        'CV_Accuracy_std': np.std(acc_scores),
        'CV_F1_mean': np.mean(f1_scores),
        'CV_F1_std': np.std(f1_scores),
        'CV_Balanced_Accuracy_mean': np.mean(bal_acc_scores),
        'CV_Balanced_Accuracy_std': np.std(bal_acc_scores),
        'CV_LogLoss_mean': np.mean(logloss_scores),
        'CV_LogLoss_std': np.std(logloss_scores)
    }
    
    results['Training_Time'] = time.time() - start_time
    results['Feature_Set'] = feature_set_name
    results['Features'] = feature_set
    
    return results
# ------------------ 4. Run All Models ------------------
model_results = {}

print("🧪 Starting model evaluation...")

# XGBoost models
model_results['XGBoost_All'] = evaluate_xgboost('All_Features', feature_sets['All_Features'])
model_results['XGBoost_NoTop3_Reg'] = evaluate_xgboost('No_Top3_Reg', feature_sets['No_Top3_Reg'])
model_results['XGBoost_NoTop3_Cls'] = evaluate_xgboost('No_Top3_Cls', feature_sets['No_Top3_Cls'])

# SVM model
model_results['SVM_All'] = evaluate_svm()

# Random Forest models
model_results['RF_All'] = evaluate_random_forest('All_Features', feature_sets['All_Features'])
model_results['RF_NoTop3_Reg'] = evaluate_random_forest('No_Top3_Reg', feature_sets['No_Top3_Reg'])
model_results['RF_NoTop3_Cls'] = evaluate_random_forest('No_Top3_Cls', feature_sets['No_Top3_Cls'])

# Create top 6 correlated features for Random Forest
top6_corr_reg = correlations_reg.abs().sort_values(ascending=False).head(6).index.tolist()
top6_corr_cls = correlations_cls.abs().sort_values(ascending=False).head(6).index.tolist()

feature_sets['No_Top6_Reg'] = [f for f in features if f not in top6_corr_reg]
feature_sets['No_Top6_Cls'] = [f for f in features if f not in top6_corr_cls]

model_results['RF_NoTop6_Reg'] = evaluate_random_forest('No_Top6_Reg', feature_sets['No_Top6_Reg'])
model_results['RF_NoTop6_Cls'] = evaluate_random_forest('No_Top6_Cls', feature_sets['No_Top6_Cls'])

# ------------------ 5. Create Combined Visualizations ------------------
print("\n📊 Creating combined visualizations...")

# Set up the plot style
plt.style.use('default')
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Model Performance Comparison for Future Forecasting', fontsize=16, fontweight='bold')

# Define the new model names mapping
rename_map = {
    "XGBoost_All": "XGBoost Lenient",
    "XGBoost_NoTop3_Reg": "XGBoost Strict",
    "SVM_All": "SVM-SVR",
    "RF_All": "Random Forest Lenient",
    "RF_NoTop3_Reg": "Random Forest Strict",
    "RF_NoTop6_Reg": "Random Forest Stricter"
}

# Colors for each model type
colors = {
    'XGBoost Lenient': '#1f77b4', 
    'XGBoost Strict': '#aec7e8',
    'SVM-SVR': '#ff7f0e', 
    'Random Forest Lenient': '#2ca02c',
    'Random Forest Strict': '#98df8a',
    'Random Forest Stricter': '#4caf50'
}

# All metrics to plot
metrics = [
    ('Regression', 'R2', 'R² Score'),
    ('Regression', 'MAE', 'MAE'),
    ('Regression', 'RMSE', 'RMSE'),
    ('Classification', 'Accuracy', 'Accuracy'),
    ('Classification', 'F1', 'F1 Score'),
    ('Classification', 'Balanced_Accuracy', 'Balanced Accuracy'),
    ('Classification', 'LogLoss', 'Log Loss'),
    ('Training_Time', 'Training_Time', 'Training Time (s)')
]

# Get renamed models
models = [rename_map[model] for model in model_results.keys() if model in rename_map]

# Plot all metrics
for i, (category, metric, title) in enumerate(metrics):
    row = i // 4
    col = i % 4
    
    values = []
    for orig_model_name in model_results.keys():
        if orig_model_name not in rename_map:
            continue
            
        if category == 'Training_Time':
            values.append(model_results[orig_model_name][metric])
        else:
            values.append(model_results[orig_model_name][category][metric])
    
    # Handle N/A values for SVM LogLoss
    if metric == 'LogLoss':
        values = [v if v != 'N/A' else 0 for v in values]
    
    bars = axes[row, col].bar(range(len(models)), values, color=[colors[model] for model in models])
    axes[row, col].set_title(title)
    axes[row, col].set_ylabel(metric if category != 'Training_Time' else 'Seconds')
    axes[row, col].set_xticks(range(len(models)))
    axes[row, col].set_xticklabels(models, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        if metric == 'LogLoss' and value == 0:  # Handle N/A for SVM LogLoss
            axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               'N/A', ha='center', va='bottom', fontsize=8)
        else:
            axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("combined_model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ------------------ 6. Detailed Results Summary ------------------
print("\n" + "="*80)
print("📋 DETAILED RESULTS SUMMARY")
print("="*80)

for orig_model_name, results in model_results.items():
    if orig_model_name not in rename_map:
        continue
        
    model_name = rename_map[orig_model_name]
    print(f"\n{model_name}:")
    print(f"  Feature Set: {results['Feature_Set']}")
    print(f"  Features: {results['Features']}")
    print(f"  Training Time: {results['Training_Time']:.2f} seconds")
    print("  Regression:")
    for metric, value in results['Regression'].items():
        if 'CV_' in metric:
            continue  # Skip CV metrics for this summary
        print(f"    {metric}: {value:.4f}")
    print("  Classification:")
    for metric, value in results['Classification'].items():
        if 'CV_' in metric:
            continue  # Skip CV metrics for this summary
        if value != 'N/A':
            print(f"    {metric}: {value:.4f}")
        else:
            print(f"    {metric}: {value}")

# Print CV results for Random Forest models
print("\n" + "="*80)
print("📊 RANDOM FOREST CROSS-VALIDATION RESULTS")
print("="*80)

rf_models = [name for name in model_results.keys() if name.startswith('RF_') and name in rename_map]
for orig_model_name in rf_models:
    model_name = rename_map[orig_model_name]
    print(f"\n{model_name}:")
    print("Regression CV Results:")
    print(f"  R²: {model_results[orig_model_name]['Regression']['CV_R2_mean']:.4f} ± {model_results[orig_model_name]['Regression']['CV_R2_std']:.4f}")
    print(f"  MAE: {model_results[orig_model_name]['Regression']['CV_MAE_mean']:.4f} ± {model_results[orig_model_name]['Regression']['CV_MAE_std']:.4f}")
    print(f"  RMSE: {model_results[orig_model_name]['Regression']['CV_RMSE_mean']:.4f} ± {model_results[orig_model_name]['Regression']['CV_RMSE_std']:.4f}")

    print("Classification CV Results:")
    print(f"  Accuracy: {model_results[orig_model_name]['Classification']['CV_Accuracy_mean']:.4f} ± {model_results[orig_model_name]['Classification']['CV_Accuracy_std']:.4f}")
    print(f"  F1: {model_results[orig_model_name]['Classification']['CV_F1_mean']:.4f} ± {model_results[orig_model_name]['Classification']['CV_F1_std']:.4f}")
    print(f"  Balanced Accuracy: {model_results[orig_model_name]['Classification']['CV_Balanced_Accuracy_mean']:.4f} ± {model_results[orig_model_name]['Classification']['CV_Balanced_Accuracy_std']:.4f}")
    print(f"  Log Loss: {model_results[orig_model_name]['Classification']['CV_LogLoss_mean']:.4f} ± {model_results[orig_model_name]['Classification']['CV_LogLoss_std']:.4f}")

# Save results to CSV
results_list = []
for orig_model_name, results in model_results.items():
    if orig_model_name not in rename_map:
        continue
        
    model_name = rename_map[orig_model_name]
    row = {
        'Model': model_name,
        'Feature_Set': results['Feature_Set'],
        'Training_Time': results['Training_Time'],
        'R2': results['Regression']['R2'],
        'MAE': results['Regression']['MAE'],
        'RMSE': results['Regression']['RMSE'],
        'Accuracy': results['Classification']['Accuracy'],
        'F1': results['Classification']['F1'],
        'Balanced_Accuracy': results['Classification']['Balanced_Accuracy'],
        'LogLoss': results['Classification']['LogLoss'] if results['Classification']['LogLoss'] != 'N/A' else np.nan
    }
    results_list.append(row)

results_df = pd.DataFrame(results_list)
results_df.to_csv("model_comparison_results.csv", index=False)
print("\n✅ Detailed results saved to 'model_comparison_results.csv'")

# ------------------ 7. Final Analysis ------------------
print("\n" + "="*80)
print("🔍 FINAL ANALYSIS")
print("="*80)

# Find best model for each metric
metrics = ['R2', 'MAE', 'RMSE', 'Accuracy', 'F1', 'Balanced_Accuracy', 'LogLoss', 'Training_Time']
best_models = {}

for metric in metrics:
    if metric == 'LogLoss':
        # For LogLoss, lower is better
        best_value = float('inf')
        best_model = None
        for orig_model_name in model_results.keys():
            if orig_model_name not in rename_map:
                continue
                
            value = model_results[orig_model_name]['Classification'][metric]
            if value != 'N/A' and value < best_value:
                best_value = value
                best_model = rename_map[orig_model_name]
    elif metric == 'Training_Time':
        # For training time, lower is better
        best_value = float('inf')
        best_model = None
        for orig_model_name in model_results.keys():
            if orig_model_name not in rename_map:
                continue
                
            value = model_results[orig_model_name][metric]
            if value < best_value:
                best_value = value
                best_model = rename_map[orig_model_name]
    elif metric in ['MAE', 'RMSE']:
        # For these regression metrics, lower is better
        best_value = float('inf')
        best_model = None
        for orig_model_name in model_results.keys():
            if orig_model_name not in rename_map:
                continue
                
            value = model_results[orig_model_name]['Regression'][metric]
            if value < best_value:
                best_value = value
                best_model = rename_map[orig_model_name]
    else:
        # For other metrics, higher is better
        best_value = -float('inf')
        best_model = None
        for orig_model_name in model_results.keys():
            if orig_model_name not in rename_map:
                continue
                
            if metric in model_results[orig_model_name]['Regression']:
                value = model_results[orig_model_name]['Regression'][metric]
            else:
                value = model_results[orig_model_name]['Classification'][metric]
            
            if value != 'N/A' and value > best_value:
                best_value = value
                best_model = rename_map[orig_model_name]
    
    best_models[metric] = (best_model, best_value)

print("\n🏆 Best Models for Each Metric:")
for metric, (model, value) in best_models.items():
    print(f"  {metric}: {model} ({value:.4f})")

print("\n✅ All models evaluated and compared successfully!")