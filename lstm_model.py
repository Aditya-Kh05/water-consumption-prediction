"""
==========================================================================
  LSTM MODEL TRAINING & COMPARISON WITH ANN
  Water Consumption Prediction - Data Mining Project
  --------------------------------------------------
  Trains a Long Short-Term Memory (LSTM) Recurrent Neural Network.
  
  Approach: Create sliding-window sequences per zone-class group,
  including the target (past demand) as an input feature so the
  LSTM can learn autoregressive temporal patterns.
  
  Key difference from ANN: LSTM sees the HISTORY of demand + weather
  over the past N months, not just the current month's snapshot.
==========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

LOOKBACK = 6  # months of history per sequence

print("=" * 65)
print("  LSTM MODEL TRAINING & EVALUATION")
print("  Water Consumption Prediction")
print("=" * 65)

# =============================================================
# 1. LOAD DATA
# =============================================================
print("\n[1/6] Loading cleaned dataset...")
df = pd.read_csv('cleaned_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Zone_Encoded', 'Class_Encoded', 'Date']).reset_index(drop=True)
print(f"  → {len(df):,} rows loaded")

# =============================================================
# 2. BUILD SEQUENCES WITH LAGGED TARGET
# =============================================================
print(f"\n[2/6] Building sequences (lookback = {LOOKBACK} months)...")

# Features for each timestep: weather + time + PAST demand
# Including past demand as a feature is key for LSTM — it makes
# the model autoregressive (similar to ARIMA but with deep learning)
input_features = [
    'Month_Sin', 'Month_Cos',
    'Avg_Temp_F', 'Total_Precip_Inches', 'Avg_Humidity',
    'Water_Demand_Gallons'   # ← past demand as input feature
]
target = 'Water_Demand_Gallons'

# Group by Zone + Customer Class
df['Group'] = df['Zone_Encoded'].astype(str) + '_' + df['Class_Encoded'].astype(str)

X_sequences = []
y_targets = []
date_indices = []  # track dates for chronological splitting
group_count = 0

for group_name, group_df in df.groupby('Group'):
    group_df = group_df.sort_values('Date').reset_index(drop=True)
    
    if len(group_df) < LOOKBACK + 1:
        continue
    
    feat_vals = group_df[input_features].values
    target_vals = group_df[target].values
    dates = group_df['Date'].values
    
    for i in range(LOOKBACK, len(group_df)):
        X_sequences.append(feat_vals[i - LOOKBACK:i])
        y_targets.append(target_vals[i])
        date_indices.append(dates[i])
    
    group_count += 1

X_sequences = np.array(X_sequences)
y_targets = np.array(y_targets)
date_indices = np.array(date_indices)

# Sort everything by date for proper chronological split
sort_idx = np.argsort(date_indices)
X_sequences = X_sequences[sort_idx]
y_targets = y_targets[sort_idx]
date_indices = date_indices[sort_idx]

print(f"  → Groups processed: {group_count}")
print(f"  → Total sequences: {X_sequences.shape[0]:,}")
print(f"  → Sequence shape: {X_sequences.shape} (samples, {LOOKBACK} timesteps, {len(input_features)} features)")
print(f"  → Features per timestep: {input_features}")

# =============================================================
# 3. CHRONOLOGICAL SPLIT & SCALING
# =============================================================
print("\n[3/6] Splitting & scaling...")

split_idx = int(len(X_sequences) * 0.8)
X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]

print(f"  → Training: {len(X_train):,} sequences")
print(f"  → Testing:  {len(X_test):,} sequences")

# Scale features — flatten to 2D, scale, reshape back to 3D
n_train, n_steps, n_feat = X_train.shape
n_test = X_test.shape[0]

scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train.reshape(-1, n_feat)).reshape(n_train, n_steps, n_feat)
X_test_s = scaler_X.transform(X_test.reshape(-1, n_feat)).reshape(n_test, n_steps, n_feat)

# Scale target
scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

os.makedirs('models', exist_ok=True)
joblib.dump(scaler_X, 'models/lstm_scaler_X.pkl')
joblib.dump(scaler_y, 'models/lstm_scaler_y.pkl')
print(f"  → Scalers saved")

# =============================================================
# 4. BUILD & TRAIN LSTM
# =============================================================
print(f"\n[4/6] Building LSTM model...")
print(f"  Architecture: BiLSTM(128) → LSTM(64) → Dense(32) → Dense(1)")

model_lstm = Sequential([
    # Bidirectional LSTM — reads sequences forwards AND backwards
    Bidirectional(LSTM(128, return_sequences=True),
                  input_shape=(LOOKBACK, n_feat)),
    Dropout(0.2),

    # Second LSTM layer
    LSTM(64, return_sequences=False),
    Dropout(0.2),

    # Dense layers
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model_lstm.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model_lstm.summary()

print("\n  Training LSTM...\n")
early_stop = EarlyStopping(monitor='val_loss', patience=20,
                           restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=8, min_lr=1e-6, verbose=1)

history = model_lstm.fit(
    X_train_s, y_train_s,
    validation_split=0.15,
    epochs=200,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =============================================================
# 5. EVALUATE
# =============================================================
print("\n[5/6] Evaluating LSTM...")

pred_s = model_lstm.predict(X_test_s, verbose=0).flatten()
lstm_predictions = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()
lstm_predictions = np.maximum(lstm_predictions, 0)

lstm_mae = mean_absolute_error(y_test, lstm_predictions)
lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_predictions))
lstm_r2 = r2_score(y_test, lstm_predictions)

print(f"\n  LSTM Results:")
print(f"    MAE:  {lstm_mae:>14,.0f} gallons")
print(f"    RMSE: {lstm_rmse:>14,.0f} gallons")
print(f"    R²:   {lstm_r2:>14.4f}")

# =============================================================
# 6. COMPARISON & CHARTS
# =============================================================
print("\n" + "=" * 65)
print("  LSTM vs ANN COMPARISON")
print("=" * 65)

# Load ANN metrics
ann_r2, ann_mae, ann_rmse = 0.8930, 3_332_922, 6_754_624
try:
    ann_metrics = joblib.load('models/comparison_metrics.pkl')
    ann_r2 = ann_metrics['ann']['R²']
    ann_mae = ann_metrics['ann']['MAE']
    ann_rmse = ann_metrics['ann']['RMSE']
    print("  (Loaded ANN results from previous training)")
except:
    print("  (Using default ANN results)")

print(f"\n  {'Metric':<20} {'LSTM':>18} {'ANN':>18}")
print("  " + "-" * 58)
print(f"  {'MAE (Gallons)':<20} {lstm_mae:>18,.0f} {ann_mae:>18,.0f}")
print(f"  {'RMSE (Gallons)':<20} {lstm_rmse:>18,.0f} {ann_rmse:>18,.0f}")
print(f"  {'R² Score':<20} {lstm_r2:>18.4f} {ann_r2:>18.4f}")

if lstm_r2 > ann_r2:
    winner = "LSTM"
    print(f"\n  ★ LSTM wins! (R² higher by {lstm_r2 - ann_r2:.4f})")
else:
    winner = "ANN"
    print(f"\n  ★ ANN still leads (R² higher by {ann_r2 - lstm_r2:.4f})")

# ---- Charts ----
print("\n[6/6] Generating charts...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax1 = axes[0]
ax1.plot(history.history['loss'], label='Training Loss', color='#9C27B0', linewidth=1.5)
ax1.plot(history.history['val_loss'], label='Validation Loss', color='#FF5722', linewidth=1.5)
ax1.set_title('LSTM Training History', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()

ax2 = axes[1]
x = np.arange(2)
width = 0.35
bars1 = ax2.bar(x - width/2, [lstm_mae, lstm_rmse], width, label='LSTM', color='#9C27B0', edgecolor='white')
bars2 = ax2.bar(x + width/2, [ann_mae, ann_rmse], width, label='ANN', color='#FF9800', edgecolor='white')
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

ax3 = axes[2]
names = ['LSTM', 'ANN\n(One-Hot Tuned)']
scores = [lstm_r2, ann_r2]
colors = ['#9C27B0', '#FF9800']
bars = ax3.bar(names, scores, color=colors, edgecolor='white', width=0.5)
ax3.set_title('R² Score Comparison', fontsize=13, fontweight='bold')
ax3.set_ylabel('R² Score')
ax3.set_ylim(0, 1.1)
for bar, score in zip(bars, scores):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
             f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('LSTM_vs_ANN_Comparison.png')
plt.close()
print("  → Saved: LSTM_vs_ANN_Comparison.png")

# Forecast chart
n_show = min(200, len(y_test))
idx = range(n_show)
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(idx, y_test[:n_show], color='black', linewidth=1.2, label='Actual', alpha=0.8)
ax.plot(idx, lstm_predictions[:n_show], color='#9C27B0', linewidth=1.2,
        linestyle='--', label='LSTM Predicted', alpha=0.8)
ax.set_title('LSTM Forecast — Actual vs Predicted', fontsize=14, fontweight='bold')
ax.set_xlabel('Test Sample Index')
ax.set_ylabel('Water Demand (Gallons)')
ax.legend()
ax.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.savefig('LSTM_Forecast.png')
plt.close()
print("  → Saved: LSTM_Forecast.png")

# Save model
model_lstm.save('models/lstm_model.keras')
joblib.dump({
    'model_name': 'LSTM (Long Short-Term Memory)',
    'mae': lstm_mae, 'rmse': lstm_rmse, 'r2': lstm_r2,
    'lookback': LOOKBACK,
    'seq_features': input_features,
    'winner': winner
}, 'models/lstm_metrics.pkl')
print("  → Saved: models/lstm_model.keras")
print("  → Saved: models/lstm_metrics.pkl")

print("\n" + "=" * 65)
print("  LSTM TRAINING COMPLETE")
print("=" * 65)
print(f"\n  LSTM R²: {lstm_r2:.4f}  |  ANN R²: {ann_r2:.4f}  |  Winner: {winner}")
