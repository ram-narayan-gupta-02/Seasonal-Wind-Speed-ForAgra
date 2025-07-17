# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# # Load the dataset
# df = pd.read_csv("formatted_wind_data.csv")

# # ---------------- Step 1: Drop missing/null rows ----------------
# df = df.dropna()

# # ---------------- Step 2: Remove duplicate rows ----------------
# df = df.drop_duplicates()

# # ---------------- Step 3: Ensure correct data types ----------------
# df[['year', 'month', 'day']] = df[['year', 'month', 'day']].astype(int)
# df[['lat', 'lon', 'altitude_m', 'wind_speed']] = df[['lat', 'lon', 'altitude_m', 'wind_speed']].astype(float)

# # ---------------- Step 4: Outlier Removal (based on z-score) ----------------
# from scipy import stats
# z_scores = stats.zscore(df[['wind_speed', 'altitude_m']])
# abs_z_scores = abs(z_scores)
# filtered_entries = (abs_z_scores < 3).all(axis=1)
# df = df[filtered_entries]


# # ---------------- Save Processed Data ----------------
# df.to_csv("preprocessed_wind_data.csv", index=False)
# print("✅ Preprocessing complete. File saved as 'preprocessed_wind_data.csv'")



# import pandas as pd

# # Load cleaned dataset
# df = pd.read_csv("preprocessed_wind_data.csv")

# # Define a function to map months to seasons
# def get_season(month):
#     if month in [3, 4, 5, 6]:
#         return "Summer"
#     elif month in [7, 8, 9, 10]:
#         return "Monsoon"
#     elif month in [11, 12, 1, 2]:
#         return "Winter"

# # Apply the function to create a 'season' column
# df['season'] = df['month'].apply(get_season)

# # Check the result
# print(df['season'].value_counts())

# # Save season-wise datasets (optional)
# df[df['season'] == 'Summer'].to_csv("wind_data_summer.csv", index=False)
# df[df['season'] == 'Monsoon'].to_csv("wind_data_monsoon.csv", index=False)
# df[df['season'] == 'Winter'].to_csv("wind_data_winter.csv", index=False)

# print("✅ Season-wise CSVs saved: wind_data_summer.csv, wind_data_monsoon.csv, wind_data_winter.csv")




import pandas as pd
import os

# Define your paths
base_path = r"D:\Python\new\Agra\data"
summer_file = os.path.join(base_path, "Summer", "wind_data_summer.csv")
monsoon_file = os.path.join(base_path, "Mansoon", "wind_data_monsoon.csv")
winter_file = os.path.join(base_path, "Winter", "wind_data_winter.csv")

# Load each season file
df_summer = pd.read_csv(summer_file)
df_monsoon = pd.read_csv(monsoon_file)
df_winter = pd.read_csv(winter_file)

# Combine them
combined_df = pd.concat([df_summer, df_monsoon, df_winter], ignore_index=True)

# Optional: Shuffle the rows
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new file
output_path = os.path.join(base_path, "wind_data_combined_seasons.csv")
combined_df.to_csv(output_path, index=False)

print(f"✅ Combined dataset saved at:\n{output_path}")