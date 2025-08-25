import xgboost as xgb
import numpy as np

# Tiny dummy dataset
X = np.random.rand(1000, 10)
y = np.random.rand(1000)

# DMatrix format for XGBoost
dtrain = xgb.DMatrix(X, label=y)

# Train with GPU
params = {
    "tree_method": "hist",   # always 'hist' in latest XGBoost
    "device": "cuda",        # force GPU
    "verbosity": 2           # show detailed logs
}

print("Training on GPU (RTX 4050 test)...")
bst = xgb.train(params, dtrain, num_boost_round=10)
print("✅ Training completed")

# Check booster attributes
print("\nBooster attributes:")
print(bst.attributes())
