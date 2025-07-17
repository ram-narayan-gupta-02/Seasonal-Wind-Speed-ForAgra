import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------ Step 1: Load Data ------------------
data_path = r"D:\Python\new\Agra\data\wind_data_combined_seasons.csv"
df = pd.read_csv(data_path)

# ------------------ Step 2: Preprocess ------------------
# One-hot encode season
if 'season' in df.columns:
    df = pd.get_dummies(df, columns=['season'])

X = df.drop(columns=['wind_speed'])
y = df['wind_speed']

# ------------------ Step 3: Split Data ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Step 4: Define Model ------------------
model = lgb.LGBMRegressor(random_state=42)

# ------------------ Step 5: Hyperparameter Tuning ------------------
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100],
    'max_depth': [-1, 10, 20],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=25,
    scoring='r2',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# ------------------ Step 6: Evaluate ------------------
y_pred = random_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ------------------ Step 7: Save Model and Metrics ------------------
model_path = r"D:\Python\new\Agra\Tuned\seasonal_lgbm_model_tuned.pkl"
metrics_path = r"D:\Python\new\Agra\Tuned\seasonal_lgbm_metrics_tuned.txt"

with open(model_path, 'wb') as f:
    pickle.dump(random_search.best_estimator_, f)

with open(metrics_path, 'w') as f:
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"MAE: {mae:.4f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write("\nBest Parameters:\n")
    f.write(str(random_search.best_params_))

print("✅ Tuned model and metrics saved successfully!")
