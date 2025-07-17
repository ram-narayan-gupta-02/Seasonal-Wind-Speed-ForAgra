import pickle
import pandas as pd

# Load the model
with open("Agra/Tuned/seasonal_lgbm_model_tuned.pkl", "rb") as f:
    model = pickle.load(f)

# Predict on new data (make sure the columns match training set!)
new_data = pd.DataFrame({
  "latitude": [27.5],
  "longitude": [77.5],
  "year": [2025],
  "month": [7],
  "day": [5],
  "altitude(m)": [110.9],
  "season_summer": [0],
  "season_monsoon": [1],
  "season_winter": [0]
}
)

prediction = model.predict(new_data)[0]
print(f"Predicted wind speed: {prediction:.2f} m/s")