"""
==========================================================================
  DATA PREPROCESSING MODULE
  Water Consumption Prediction - Data Mining Project
  --------------------------------------------------
  This script handles all data cleaning, transformation, and feature
  engineering. It produces a clean dataset ready for EDA and modeling.
==========================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("=" * 60)
print("  STEP 1: DATA PREPROCESSING & FEATURE ENGINEERING")
print("=" * 60)

# ---------------------------------------------------------
# 1. Load the raw merged dataset
# ---------------------------------------------------------
print("\n[1/6] Loading raw dataset...")
df = pd.read_csv('Zone_Level_Water_Weather_Merged.csv')
print(f"  → Loaded {len(df):,} rows and {len(df.columns)} columns.")
print(f"  → Columns: {list(df.columns)}")

# ---------------------------------------------------------
# 2. Fix Water_Demand_Gallons (remove commas, convert to float)
# ---------------------------------------------------------
print("\n[2/6] Cleaning 'Water_Demand_Gallons' column...")
df['Water_Demand_Gallons'] = (
    df['Water_Demand_Gallons']
    .astype(str)
    .str.replace(',', '', regex=False)
    .astype(float)
)
print(f"  → Demand range: {df['Water_Demand_Gallons'].min():,.0f} to {df['Water_Demand_Gallons'].max():,.0f} gallons")

# Remove rows with zero demand (they add noise and don't represent real consumption)
rows_before = len(df)
df = df[df['Water_Demand_Gallons'] > 0].copy()
print(f"  → Removed {rows_before - len(df):,} rows with zero demand. Remaining: {len(df):,} rows.")

# ---------------------------------------------------------
# 3. Create proper Date column
# ---------------------------------------------------------
print("\n[3/6] Creating Date column...")
df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
df = df.sort_values('Date').reset_index(drop=True)

# ---------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------
print("\n[4/6] Engineering new features...")

# 4a. Cyclical encoding of Month (preserves Jan close to Dec relationship)
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# 4b. Season as a categorical variable
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['Season'] = df['Month'].apply(get_season)
print(f"  → Created: Month_Sin, Month_Cos, Season")
print(f"  → Season distribution:\n{df['Season'].value_counts().to_string()}")

# ---------------------------------------------------------
# 5. Encode categorical variables
# ---------------------------------------------------------
print("\n[5/6] Encoding categorical variables...")

# Clean up Zone_ZipCode (some have suffixes like '78703-1834')
df['Zone_ZipCode'] = df['Zone_ZipCode'].astype(str).str[:5]

# Label Encode Zone_ZipCode
le_zone = LabelEncoder()
df['Zone_Encoded'] = le_zone.fit_transform(df['Zone_ZipCode'])
print(f"  → Zone_ZipCode: {len(le_zone.classes_)} unique zones encoded")

# Label Encode Customer Class
le_class = LabelEncoder()
df['Class_Encoded'] = le_class.fit_transform(df['Customer Class'])
print(f"  → Customer Class: {list(le_class.classes_)}")

# Label Encode Season
le_season = LabelEncoder()
df['Season_Encoded'] = le_season.fit_transform(df['Season'])

# Save the encoders (needed by the web app for prediction)
os.makedirs('models', exist_ok=True)
joblib.dump(le_zone, 'models/label_encoder_zone.pkl')
joblib.dump(le_class, 'models/label_encoder_class.pkl')
joblib.dump(le_season, 'models/label_encoder_season.pkl')
print("  → Saved label encoders to models/ directory")

# ---------------------------------------------------------
# 6. Save cleaned dataset
# ---------------------------------------------------------
print("\n[6/6] Saving cleaned dataset...")
df.to_csv('cleaned_data.csv', index=False)
print(f"  → Saved 'cleaned_data.csv' ({len(df):,} rows, {len(df.columns)} columns)")

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("  PREPROCESSING COMPLETE")
print("=" * 60)
print(f"\n  Final columns: {list(df.columns)}")
print(f"\n  Dataset shape: {df.shape}")
print(f"  Date range: {df['Date'].min().strftime('%b %Y')} to {df['Date'].max().strftime('%b %Y')}")
print(f"  Unique Zones: {df['Zone_ZipCode'].nunique()}")
print(f"  Customer Classes: {list(le_class.classes_)}")
print(f"\n  Files created:")
print(f"    - cleaned_data.csv")
print(f"    - models/label_encoder_zone.pkl")
print(f"    - models/label_encoder_class.pkl")
print(f"    - models/label_encoder_season.pkl")
