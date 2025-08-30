

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
from prophet import Prophet
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
    
    # Store models for later forecasting
    results['Regression_Model'] = model_reg
    results['Classification_Model'] = model_cls
    
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
    
    # Store models for later forecasting
    results['Regression_Model'] = svr_model
    results['Classification_Model'] = svc_model
    
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
    
    # Store models for later forecasting
    results['Regression_Model'] = rf_reg_final
    results['Classification_Model'] = rf_cls_final
    
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

# ------------------ 5. Prophet Forecasting for All Features and Targets ------------------
print("\n🔮 Forecasting with Prophet for all features and targets...")

# Prepare data for Prophet
if not date_col:
    # Create a date column if none exists
    df_clean['date'] = pd.date_range(start='2000-01-01', periods=len(df_clean), freq='D')
    date_col = 'date'

# Dictionary to store all forecasts
prophet_forecasts = {}

# Features and targets to forecast
all_columns = features + ['CCME_Values', 'CCME_WQI']

# For each column, create a Prophet model and forecast
for column in all_columns:
    print(f"📊 Forecasting {column}...")
    
    # Prepare data for Prophet
    prophet_df = df_clean[[date_col, column]].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Create and fit Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0
    )
    
    model.fit(prophet_df)
    
    # Make future dataframe for 5 years (365*5 days)
    future = model.make_future_dataframe(periods=5, freq='Y')
    
    # Forecast
    forecast = model.predict(future)
    prophet_forecasts[column] = forecast
    
    print(f"✅ Completed forecasting for {column}")

# ------------------ 6. Create ML Model Forecasts Using Prophet Features ------------------
print("\n🤖 Creating ML model forecasts using Prophet features...")

# Prepare future feature data from Prophet forecasts
future_features = pd.DataFrame()
future_features['ds'] = prophet_forecasts[features[0]]['ds']

for feature in features:
    future_features[feature] = prophet_forecasts[feature]['yhat']

# Filter to only future dates (last 5 years)
last_historical_date = df_clean[date_col].max()
future_features = future_features[future_features['ds'] > last_historical_date]

# Dictionary to store ML model forecasts
ml_forecasts = {}

# Define model names mapping
rename_map = {
    "XGBoost_All": "XGBoost Lenient",
    "XGBoost_NoTop3_Reg": "XGBoost Strict",
    "SVM_All": "SVM-SVR",
    "RF_All": "Random Forest Lenient",
    "RF_NoTop3_Reg": "Random Forest Strict",
    "RF_NoTop6_Reg": "Random Forest Stricter"
}

# For each model, forecast the targets using the Prophet feature forecasts
for model_key, model_name in rename_map.items():
    print(f"📊 Creating forecasts for {model_name}...")
    
    # Get the model results
    model_result = model_results[model_key]
    
    # Get the feature set used by this model
    feature_set = model_result['Features']
    
    # Prepare future data with the correct features
    future_X = future_features[feature_set]
    
    # Forecast regression target (CCME_Values)
    if hasattr(model_result['Regression_Model'], 'predict'):
        # For scikit-learn models
        reg_forecast = model_result['Regression_Model'].predict(future_X)
    else:
        # For XGBoost models
        dfuture = xgb.DMatrix(future_X)
        reg_forecast = model_result['Regression_Model'].predict(dfuture)
    
    # Forecast classification target (CCME_WQI)
    if hasattr(model_result['Classification_Model'], 'predict'):
        # For scikit-learn models
        cls_forecast = model_result['Classification_Model'].predict(future_X)
    else:
        # For XGBoost models
        dfuture = xgb.DMatrix(future_X)
        cls_proba = model_result['Classification_Model'].predict(dfuture)
        cls_forecast = np.argmax(cls_proba, axis=1)
    
    # Store the forecasts
    ml_forecasts[model_name] = {
        'CCME_Values': reg_forecast,
        'CCME_WQI': cls_forecast,
        'dates': future_features['ds']
    }
    
    print(f"✅ Completed forecasts for {model_name}")

# ------------------ 7. Create Combined Visualizations ------------------
print("\n🎨 Creating combined visualizations...")

# Set up the plot style
plt.style.use('default')
plt.rcParams['figure.figsize'] = [15, 10]

# Colors for each model
model_colors = {
    'XGBoost Lenient': '#1f77b4',
    'XGBoost Strict': '#aec7e8',
    'SVM-SVR': '#ff7f0e',
    'Random Forest Lenient': '#2ca02c',
    'Random Forest Strict': '#98df8a',
    'Random Forest Stricter': '#4caf50',
    'Prophet': '#7f7f7f'
}

# Create subplots for each target
fig, axes = plt.subplots(2, 1, figsize=(15, 12))
fig.suptitle('5-Year Forecast Comparison for Water Quality Targets', fontsize=16, fontweight='bold')

# Plot CCME_Values forecasts
ax1 = axes[0]
# Plot Prophet forecast
prophet_dates = prophet_forecasts['CCME_Values']['ds']
prophet_values = prophet_forecasts['CCME_Values']['yhat']
future_mask = prophet_dates > last_historical_date
ax1.plot(prophet_dates[future_mask], prophet_values[future_mask], 
         label='Prophet', color=model_colors['Prophet'], linewidth=3)

# Plot ML model forecasts
for model_name, forecast in ml_forecasts.items():
    ax1.plot(forecast['dates'], forecast['CCME_Values'], 
             label=model_name, color=model_colors[model_name], linewidth=2, linestyle='--')

ax1.set_title('CCME_Values Forecast')
ax1.set_ylabel('CCME_Values')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot CCME_WQI forecasts
ax2 = axes[1]
# Plot Prophet forecast
prophet_dates = prophet_forecasts['CCME_WQI']['ds']
prophet_values = prophet_forecasts['CCME_WQI']['yhat']
future_mask = prophet_dates > last_historical_date
ax2.plot(prophet_dates[future_mask], prophet_values[future_mask], 
         label='Prophet', color=model_colors['Prophet'], linewidth=3)

# Plot ML model forecasts
for model_name, forecast in ml_forecasts.items():
    ax2.plot(forecast['dates'], forecast['CCME_WQI'], 
             label=model_name, color=model_colors[model_name], linewidth=2, linestyle='--')

ax2.set_title('CCME_WQI Forecast')
ax2.set_ylabel('CCME_WQI')
ax2.set_xlabel('Date')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("combined_forecast_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# Create a grid of subplots for feature forecasts
print("\n📊 Creating feature forecast visualizations...")

n_features = len(features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
fig.suptitle('5-Year Feature Forecasts from Prophet', fontsize=16, fontweight='bold')
axes = axes.flatten()

for i, feature in enumerate(features):
    forecast_df = prophet_forecasts[feature]
    
    # Plot historical data
    historical_mask = forecast_df['ds'] <= last_historical_date
    axes[i].plot(forecast_df['ds'][historical_mask], forecast_df['yhat'][historical_mask], 
                color='blue', alpha=0.7, label='Historical Fit')
    
    # Plot forecast
    future_mask = forecast_df['ds'] > last_historical_date
    axes[i].plot(forecast_df['ds'][future_mask], forecast_df['yhat'][future_mask], 
                color='red', alpha=0.7, label='Forecast')
    
    # Add uncertainty interval
    axes[i].fill_between(forecast_df['ds'][future_mask], 
                        forecast_df['yhat_lower'][future_mask], 
                        forecast_df['yhat_upper'][future_mask], 
                        color='red', alpha=0.2)
    
    axes[i].set_title(f'{feature} Forecast')
    axes[i].set_ylabel(feature)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

# Remove any empty subplots
for i in range(n_features, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig("feature_forecasts.png", dpi=300, bbox_inches='tight')
plt.show()

# ------------------ 8. Create Summary Statistics ------------------
print("\n📈 Generating forecast summary statistics...")

# Create a summary DataFrame for the target forecasts
summary_data = []

for model_name, forecast in ml_forecasts.items():
    for target in ['CCME_Values', 'CCME_WQI']:
        values = forecast[target]
        summary_data.append({
            'Model': model_name,
            'Target': target,
            'Start_Value': values[0],
            'End_Value': values[-1],
            'Percent_Change': ((values[-1] - values[0]) / values[0]) * 100,
            'Average_Value': np.mean(values),
            'Std_Dev': np.std(values)
        })

# Add Prophet forecasts to summary
for target in ['CCME_Values', 'CCME_WQI']:
    prophet_future = prophet_forecasts[target]['ds'] > last_historical_date
    values = prophet_forecasts[target]['yhat'][prophet_future].values
    summary_data.append({
        'Model': 'Prophet',
        'Target': target,
        'Start_Value': values[0],
        'End_Value': values[-1],
        'Percent_Change': ((values[-1] - values[0]) / values[0]) * 100,
        'Average_Value': np.mean(values),
        'Std_Dev': np.std(values)
    })

# Convert to DataFrame
summary_df = pd.DataFrame(summary_data)
print("\n📊 Forecast Summary Statistics:")
print(summary_df)

# Save summary to CSV
summary_df.to_csv("forecast_summary.csv", index=False)
print("\n✅ Forecast summary saved to 'forecast_summary.csv'")

print("\n🎉 All forecasting completed successfully!")