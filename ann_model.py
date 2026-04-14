"""
==========================================================================
  ANN MODEL (TUNED v2) — TRAINING & COMPARISON WITH RANDOM FOREST
  Water Consumption Prediction - Data Mining Project
  --------------------------------------------------
  Key Fix: One-Hot Encode Zone & Customer Class so the ANN doesn't
  treat them as ordinal numbers. Scale target with StandardScaler.
  
  Tuning Log:
    v0 (baseline):  R² = 0.6796  (raw features, no target scaling)
    v1 (this):       One-hot categoricals + target scaling + tuned arch
==========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

print("=" * 65)
print("  TUNED ANN (v2) vs RANDOM FOREST — MODEL COMPARISON")
print("  Water Consumption Prediction")
print("=" * 65)

# =============================================================
# 1. LOAD DATA
# =============================================================
print("\n[1/7] Loading cleaned dataset...")
df = pd.read_csv('cleaned_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
print(f"  → {len(df):,} rows loaded")

# =============================================================
# 2. PREPARE FEATURES — THE KEY FIX
# =============================================================
print("\n[2/7] Preparing features (One-Hot Encoding categoricals)...")

# Numerical features (these are fine as-is)
num_features = ['Month_Sin', 'Month_Cos', 'Avg_Temp_F',
                'Total_Precip_Inches', 'Avg_Humidity']

# Categorical features — MUST be one-hot encoded for ANN
# (Label encoding makes ANN think Zone 40 > Zone 5, which is wrong)
cat_features = ['Zone_Encoded', 'Class_Encoded']

# Also keep the original label-encoded features for Random Forest
rf_feature_columns = [
    'Month_Sin', 'Month_Cos', 'Zone_Encoded', 'Class_Encoded',
    'Avg_Temp_F', 'Total_Precip_Inches', 'Avg_Humidity', 'Season_Encoded',
]

target = 'Water_Demand_Gallons'

# One-Hot Encode categoricals for ANN
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
cat_encoded = ohe.fit_transform(df[cat_features].values)
print(f"  → One-Hot Encoded: {cat_encoded.shape[1]} columns from {len(cat_features)} categorical features")
print(f"    (48 zones + 4 customer classes = 52 binary columns)")

# Combine: numerical + one-hot categoricals
X_ann = np.hstack([df[num_features].values, cat_encoded])
X_rf = df[rf_feature_columns].values
y = df[target].values

print(f"  → ANN input shape: {X_ann.shape[1]} features")
print(f"  → RF input shape:  {X_rf.shape[1]} features")

# Save one-hot encoder for web app
os.makedirs('models', exist_ok=True)
joblib.dump(ohe, 'models/onehot_encoder.pkl')
joblib.dump(num_features, 'models/num_features.pkl')
joblib.dump(cat_features, 'models/cat_features.pkl')

# =============================================================
# 3. CHRONOLOGICAL SPLIT
# =============================================================
print("\n[3/7] Splitting data (80/20 chronological)...")
split_idx = int(len(df) * 0.8)

X_ann_train, X_ann_test = X_ann[:split_idx], X_ann[split_idx:]
X_rf_train, X_rf_test = X_rf[:split_idx], X_rf[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"  → Training: {len(y_train):,} samples")
print(f"  → Testing:  {len(y_test):,} samples")

# =============================================================
# 4. SCALE FEATURES & TARGET
# =============================================================
print("\n[4/7] Scaling features and target...")

# Scale ANN features
scaler_X = StandardScaler()
X_ann_train_s = scaler_X.fit_transform(X_ann_train)
X_ann_test_s = scaler_X.transform(X_ann_test)

# Scale target (keeps original scale relationships, just normalizes range)
scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_s = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

joblib.dump(scaler_X, 'models/scaler.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')
print(f"  → Target mean: {scaler_y.mean_[0]:,.0f} gallons")
print(f"  → Target std:  {scaler_y.scale_[0]:,.0f} gallons")

# =============================================================
# 5. BUILD & TRAIN ANN
# =============================================================
n_inputs = X_ann_train_s.shape[1]
print(f"\n[5/7] Building ANN ({n_inputs} inputs → 512 → 256 → 128 → 64 → 1)...")

model_ann = Sequential([
    # Layer 1 — wide to handle 57 input features
    Dense(512, input_shape=(n_inputs,)),
    LeakyReLU(negative_slope=0.1),
    BatchNormalization(),
    Dropout(0.2),

    # Layer 2
    Dense(256),
    LeakyReLU(negative_slope=0.1),
    BatchNormalization(),
    Dropout(0.2),

    # Layer 3
    Dense(128),
    LeakyReLU(negative_slope=0.1),
    BatchNormalization(),
    Dropout(0.15),

    # Layer 4
    Dense(64),
    LeakyReLU(negative_slope=0.1),

    # Output
    Dense(1, activation='linear')
])

model_ann.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='mse',
    metrics=['mae']
)

model_ann.summary()

print("\n  Training ANN...\n")
early_stop = EarlyStopping(monitor='val_loss', patience=25,
                           restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=1e-7, verbose=1)

history = model_ann.fit(
    X_ann_train_s, y_train_s,
    validation_split=0.15,
    epochs=300,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Inverse-transform predictions
ann_pred_s = model_ann.predict(X_ann_test_s, verbose=0).flatten()
ann_predictions = scaler_y.inverse_transform(ann_pred_s.reshape(-1, 1)).flatten()
ann_predictions = np.maximum(ann_predictions, 0)

ann_mae = mean_absolute_error(y_test, ann_predictions)
ann_rmse = np.sqrt(mean_squared_error(y_test, ann_predictions))
ann_r2 = r2_score(y_test, ann_predictions)

print(f"\n  Tuned ANN Results:")
print(f"    MAE:  {ann_mae:>14,.0f} gallons")
print(f"    RMSE: {ann_rmse:>14,.0f} gallons")
print(f"    R²:   {ann_r2:>14.4f}")

# =============================================================
# 6. TRAIN RANDOM FOREST
# =============================================================
print("\n[6/7] Training Random Forest for comparison...")
model_rf = RandomForestRegressor(
    n_estimators=200, max_depth=20,
    min_samples_split=5, random_state=42, n_jobs=-1
)
model_rf.fit(X_rf_train, y_train)

rf_predictions = model_rf.predict(X_rf_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print(f"  Random Forest Results:")
print(f"    MAE:  {rf_mae:>14,.0f} gallons")
print(f"    RMSE: {rf_rmse:>14,.0f} gallons")
print(f"    R²:   {rf_r2:>14.4f}")

# =============================================================
# 7. COMPARISON & CHARTS
# =============================================================
print("\n" + "=" * 65)
print("  HEAD-TO-HEAD COMPARISON")
print("=" * 65)
print(f"\n  {'Metric':<20} {'Tuned ANN':>18} {'Random Forest':>18}")
print("  " + "-" * 58)
print(f"  {'MAE (Gallons)':<20} {ann_mae:>18,.0f} {rf_mae:>18,.0f}")
print(f"  {'RMSE (Gallons)':<20} {ann_rmse:>18,.0f} {rf_rmse:>18,.0f}")
print(f"  {'R² Score':<20} {ann_r2:>18.4f} {rf_r2:>18.4f}")

if ann_r2 > rf_r2:
    winner = "Tuned ANN"
    print(f"\n  ★ WINNER: Tuned ANN! (R² higher by {ann_r2 - rf_r2:.4f})")
else:
    winner = "Random Forest"
    diff = rf_r2 - ann_r2
    print(f"\n  ★ Random Forest still leads (R² higher by {diff:.4f})")

print(f"\n  ANN Improvement Journey: 0.6796 → {ann_r2:.4f}")

# ---- Charts ----
print("\n[7/7] Generating charts...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Training History
ax1 = axes[0]
ax1.plot(history.history['loss'], label='Training Loss', color='#2196F3', linewidth=1.5)
ax1.plot(history.history['val_loss'], label='Validation Loss', color='#FF5722', linewidth=1.5)
ax1.set_title('ANN Training History', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()

# Plot 2: Error bars
ax2 = axes[1]
x = np.arange(2)
width = 0.35
bars1 = ax2.bar(x - width/2, [ann_mae, ann_rmse], width, label='Tuned ANN', color='#FF9800', edgecolor='white')
bars2 = ax2.bar(x + width/2, [rf_mae, rf_rmse], width, label='Random Forest', color='#4CAF50', edgecolor='white')
ax2.set_title('Error Comparison (Lower is Better)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['MAE', 'RMSE'])
ax2.set_ylabel('Gallons')
ax2.legend()
ax2.ticklabel_format(style='plain', axis='y')
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{bar.get_height()/1e6:.1f}M', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{bar.get_height()/1e6:.1f}M', ha='center', va='bottom', fontsize=9)

# Plot 3: R² Comparison
ax3 = axes[2]
versions = ['ANN\n(Baseline)', 'ANN\n(One-Hot Tuned)', 'Random\nForest']
r2_scores = [0.6796, ann_r2, rf_r2]
colors = ['#FFCC80', '#FF9800', '#4CAF50']
bars = ax3.bar(versions, r2_scores, color=colors, edgecolor='white', width=0.5)
ax3.set_title('R² Score Comparison', fontsize=13, fontweight='bold')
ax3.set_ylabel('R² Score')
ax3.set_ylim(0, 1.1)
for bar, score in zip(bars, r2_scores):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('ANN_vs_RF_Comparison.png')
plt.close()
print("  → Saved: ANN_vs_RF_Comparison.png")

# Actual vs Predicted
n_show = min(150, len(y_test))
idx = range(n_show)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

ax1.plot(idx, y_test[:n_show], color='black', linewidth=1, label='Actual', alpha=0.8)
ax1.plot(idx, ann_predictions[:n_show], color='#FF9800', linewidth=1, linestyle='--', label='ANN Predicted', alpha=0.8)
ax1.set_title('Tuned ANN — Actual vs Predicted', fontsize=13, fontweight='bold')
ax1.set_xlabel('Test Sample')
ax1.set_ylabel('Water Demand (Gallons)')
ax1.legend()
ax1.ticklabel_format(style='plain', axis='y')

ax2.plot(idx, y_test[:n_show], color='black', linewidth=1, label='Actual', alpha=0.8)
ax2.plot(idx, rf_predictions[:n_show], color='#4CAF50', linewidth=1, linestyle='--', label='RF Predicted', alpha=0.8)
ax2.set_title('Random Forest — Actual vs Predicted', fontsize=13, fontweight='bold')
ax2.set_xlabel('Test Sample')
ax2.set_ylabel('Water Demand (Gallons)')
ax2.legend()
ax2.ticklabel_format(style='plain', axis='y')

plt.tight_layout()
plt.savefig('ANN_vs_RF_Predictions.png')
plt.close()
print("  → Saved: ANN_vs_RF_Predictions.png")

# Save model
model_ann.save('models/ann_model.keras')
print("  → Saved: models/ann_model.keras")

joblib.dump({
    'ann': {'MAE': ann_mae, 'RMSE': ann_rmse, 'R²': ann_r2},
    'rf': {'MAE': rf_mae, 'RMSE': rf_rmse, 'R²': rf_r2},
    'winner': winner,
    'tuning_journey': [0.6796, ann_r2]
}, 'models/comparison_metrics.pkl')

print("\n" + "=" * 65)
print("  COMPLETE")
print("=" * 65)
print(f"\n  ANN Tuning: 0.6796 → {ann_r2:.4f}")
