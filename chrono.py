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
    results['Models'] = {'reg': model_reg, 'cls': model_cls}

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
    results['Models'] = {'reg': svr_model, 'cls': svc_model}

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
    results['Models'] = {'reg': rf_reg_final, 'cls': rf_cls_final}

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

# ------------------ 5. Unify Results ------------------
print("\n📊 Creating unified results...")

rename_map = {
    "XGBoost_All": "XGBoost Lenient",
    "XGBoost_NoTop3_Reg": "XGBoost Strict",
    "SVM_All": "SVM–SVR",
    "RF_All": "Random Forest Lenient",
    "RF_NoTop3_Reg": "Random Forest Strict",
    "RF_NoTop6_Reg": "Random Forest Stricter"
}

# Keep only the six main models
combined_results = {}
for key, res in model_results.items():
    if key not in rename_map:
        continue
    new_key = rename_map[key]

    combined_results[new_key] = {
        'R2': res['Regression']['R2'],
        'MAE': res['Regression']['MAE'],
        'RMSE': res['Regression']['RMSE'],
        'Accuracy': res['Classification']['Accuracy'] if res['Classification']['Accuracy'] != 'N/A' else np.nan,
        'F1': res['Classification']['F1'] if res['Classification']['F1'] != 'N/A' else np.nan,
        'Balanced_Accuracy': res['Classification']['Balanced_Accuracy'] if res['Classification']['Balanced_Accuracy'] != 'N/A' else np.nan,
        'LogLoss': res['Classification']['LogLoss'] if res['Classification']['LogLoss'] != 'N/A' else np.nan,
        'Training_Time': res['Training_Time']
    }

# ------------------ 6. Unified Plots ------------------
print("\n📊 Plotting unified metrics...")

plt.style.use('default')
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Unified Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = [
    ('R2', 'R² Score'),
    ('MAE', 'MAE'),
    ('RMSE', 'RMSE'),
    ('Accuracy', 'Accuracy'),
    ('F1', 'F1 Score'),
    ('Balanced_Accuracy', 'Balanced Accuracy'),
    ('LogLoss', 'Log Loss'),
    ('Training_Time', 'Training Time (s)')
]

models = list(combined_results.keys())

for i, (metric, title) in enumerate(metrics):
    row = i // 4
    col = i % 4

    values = [combined_results[m][metric] for m in models]
    bars = axes[row, col].bar(models, values)

    axes[row, col].set_title(title)
    axes[row, col].set_ylabel(metric if metric != 'Training_Time' else 'Seconds')
    axes[row, col].set_xticklabels(models, rotation=30, ha='right')

    for bar, value in zip(bars, values):
        if pd.isna(value):
            axes[row, col].text(bar.get_x() + bar.get_width()/2., 0.01, 'N/A',
                                ha='center', va='bottom', fontsize=8)
        else:
            axes[row, col].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                                f'{value:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig("unified_model_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ------------------ 7. Save Unified Results ------------------
results_df = pd.DataFrame.from_dict(combined_results, orient='index')
results_df.to_csv("unified_model_results.csv")
print("\n✅ Unified results saved to 'unified_model_results.csv'")

# ------------------ 8. Final Analysis ------------------
print("\n" + "="*80)
print("🔍 FINAL ANALYSIS")
print("="*80)

best_models = {}
for metric in ['R2', 'MAE', 'RMSE', 'Accuracy', 'F1', 'Balanced_Accuracy', 'LogLoss', 'Training_Time']:
    values = results_df[metric]

    if metric in ['MAE', 'RMSE', 'LogLoss', 'Training_Time']:
        best_idx = values.idxmin()
        best_val = values.min()
    else:
        best_idx = values.idxmax()
        best_val = values.max()

    best_models[metric] = (best_idx, best_val)

print("\n🏆 Best Models for Each Metric:")
for metric, (model, value) in best_models.items():
    print(f"  {metric}: {model} ({value:.4f})")

print("\n✅ All models evaluated and compared successfully!")


# ------------------ 9. Forecast Future Values with Prophet ------------------
from prophet import Prophet

HORIZON_DAYS = 1800  # forecast 5 yearss ahead
print(f"\n📈 Forecasting next {HORIZON_DAYS} days...")

# Make sure date is sorted
df[date_col] = pd.to_datetime(df[date_col])
df_for_prophet = df.set_index(date_col).sort_index()

# Forecast each feature
future_index = None
future_features = {}
for f in features:
    series = df_for_prophet[[f]].dropna().resample('W').mean()  # weekly average
    series = series.reset_index().rename(columns={date_col: 'ds', f: 'y'})
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(series)
    future = m.make_future_dataframe(periods=HORIZON_DAYS, freq='D')
    forecast = m.predict(future)
    
    tail = forecast[['ds', 'yhat']].tail(HORIZON_DAYS).set_index('ds')['yhat']
    if future_index is None:
        future_index = tail.index
    future_features[f] = tail.reindex(future_index)

# Build future feature dataframe
X_future = pd.DataFrame(future_features, index=future_index)
print(f"✅ Future features generated: {X_future.shape}")

# ------------------ 10. Forecast with All 6 Models ------------------
future_preds = {}

# Helper: predict with XGBoost
def predict_xgb(trained_model, features):
    dmat = xgb.DMatrix(X_future[features])
    return trained_model.predict(dmat)

# XGBoost lenient & strict
future_preds["XGBoost Lenient"] = predict_xgb(model_results['XGBoost_All']['Models']['reg'],
                                              feature_sets['All_Features'])
future_preds["XGBoost Strict"] = predict_xgb(model_results['XGBoost_NoTop3_Reg']['Models']['reg'],
                                             feature_sets['No_Top3_Reg'])

# SVM
svr_model = model_results['SVM_All']['Models']['reg']
future_preds["SVM–SVR"] = svr_model.predict(X_future[features])

# Random Forest lenient/strict/stricter
future_preds["Random Forest Lenient"] = model_results['RF_All']['Models']['reg'].predict(X_future[feature_sets['All_Features']])
future_preds["Random Forest Strict"] = model_results['RF_NoTop3_Reg']['Models']['reg'].predict(X_future[feature_sets['No_Top3_Reg']])
future_preds["Random Forest Stricter"] = model_results['RF_NoTop6_Reg']['Models']['reg'].predict(X_future[feature_sets['No_Top6_Reg']])

# ------------------ 11. Forecast Classification (CCME_WQI) ------------------
print("\n📈 Forecasting CCME_WQI categories with all models...")

# Reverse map: 0 → Excellent, etc.
inv_label_mapping = {v: k for k, v in label_mapping.items()}

future_preds_cls = {}

# XGBoost Lenient & Strict
future_preds_cls["XGBoost Lenient"] = model_results['XGBoost_All']['Models']['cls'].predict(
    xgb.DMatrix(X_future[feature_sets['All_Features']])
).argmax(axis=1)
future_preds_cls["XGBoost Strict"] = model_results['XGBoost_NoTop3_Reg']['Models']['cls'].predict(
    xgb.DMatrix(X_future[feature_sets['No_Top3_Reg']])
).argmax(axis=1)

# SVM
svc_model = model_results['SVM_All']['Models']['cls']
future_preds_cls["SVM–SVC"] = svc_model.predict(X_future[features])

# Random Forest Lenient/Strict/Stricter
future_preds_cls["Random Forest Lenient"] = model_results['RF_All']['Models']['cls'].predict(
    X_future[feature_sets['All_Features']])
future_preds_cls["Random Forest Strict"] = model_results['RF_NoTop3_Reg']['Models']['cls'].predict(
    X_future[feature_sets['No_Top3_Reg']])
future_preds_cls["Random Forest Stricter"] = model_results['RF_NoTop6_Reg']['Models']['cls'].predict(
    X_future[feature_sets['No_Top6_Reg']])

# Convert numeric → category labels
future_preds_cls_named = {
    model: [inv_label_mapping[int(p)] for p in preds]
    for model, preds in future_preds_cls.items()
}


# ------------------ 12. Unified Forecast Dashboard ------------------
print("\n📊 Creating unified forecast dashboard...")

plt.style.use("seaborn-v0_8-colorblind")
fig, axes = plt.subplots(3, 1, figsize=(16, 18))

# --- 1. Regression Forecast (CCME_Values) ---
for name, preds in future_preds.items():
    axes[0].plot(X_future.index, preds, label=name, linewidth=2)
axes[0].set_title("Forecast of CCME_Values by All Models", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Predicted CCME_Values")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.6)

# --- 2. Classification Forecast (WQI numeric codes) ---
for name, preds in future_preds_cls.items():
    axes[1].plot(X_future.index, preds, label=name, linewidth=2)
axes[1].set_title("Forecast of CCME_WQI Classes (0=Excellent → 4=Poor)", fontsize=14, fontweight='bold')
axes[1].set_ylabel("Predicted WQI Class")
axes[1].set_yticks(range(5))
axes[1].set_yticklabels(["Excellent","Fair","Good","Marginal","Poor"])  # category labels
axes[1].legend()
axes[1].grid(True, linestyle="--", alpha=0.6)

# --- 3. Distribution of Predicted WQI Categories ---
freq_df = pd.DataFrame(future_preds_cls_named, index=X_future.index)
dist_df = freq_df.apply(lambda x: pd.Series(x).value_counts(), axis=1).fillna(0)
dist_df.plot.area(ax=axes[2], alpha=0.7, cmap="tab20c")
axes[2].set_title("Forecast Distribution of CCME_WQI Categories", fontsize=14, fontweight='bold')
axes[2].set_ylabel("Frequency")
axes[2].grid(True, linestyle="--", alpha=0.6)

# --- Final Layout ---
plt.tight_layout()
plt.savefig("forecast_dashboard.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Unified forecast dashboard saved as forecast_dashboard.png")
