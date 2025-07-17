# LGBM model

# ✅ Required Libraries
import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# ✅ Paths
DATA_PATH = r"D:\Python\new\Agra\data\wind_data_combined_seasons.csv"
MODEL_PATH = r"D:\Python\new\Agra\model\seasonal_lgbm_model.pkl"
METRIC_PATH = r"D:\Python\new\Agra\model\seasonal_lgbm_metrics.txt"

# ✅ Load Dataset
df = pd.read_csv(DATA_PATH)

# ✅ One-hot encode 'season' (categorical)
df = pd.get_dummies(df, columns=["season"], drop_first=True)

# ✅ Define Features and Target
X = df.drop(columns=['wind_speed'])
y = df['wind_speed']

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Train LightGBM Model
model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ✅ Predictions
y_pred = model.predict(X_test)

# ✅ Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ✅ Save Model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

# ✅ Save Metrics
with open(METRIC_PATH, "w", encoding="utf-8") as f:
    f.write(f"📊 Model Evaluation Metrics\n")
    f.write(f"MSE: {mse:.4f}\n")
    f.write(f"MAE: {mae:.4f} m/s\n")
    f.write(f"R² Score: {r2:.4f}\n")

print("✅ Model trained and saved successfully!")
print(f"📦 Model path: {MODEL_PATH}")
print(f"📈 Metrics path: {METRIC_PATH}")
