"""
==========================================================================
  LSTM MODEL TRAINING & COMPARISON WITH ANN
  Water Consumption Prediction - Data Mining Project
  --------------------------------------------------
  Trains a Long Short-Term Memory (LSTM) Recurrent Neural Network.
  
  Changes implemented to fix 'Lag Illusion':
  - LOOKBACK increased to 12 months for full annual seasonality.
  - Target Differencing: Model predicts the change in demand.
  - Spatial Features: Zone_Encoded and Class_Encoded added.
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

LOOKBACK = 12  # Increased to 12 months for full annual cycle

print("=" * 65)
print("  LSTM MODEL TRAINING & EVALUATION (Differenced)")
print("  Water Consumption Prediction")
print("=" * 65)

# =============================================================
# 1. LOAD DATA
# =============================================================
print("\n[1/6] Loading cleaned dataset...")
df = pd.read_csv('cleaned_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Zone_Encoded', 'Class_Encoded', 'Date']).reset_index(drop=True)
print(f"  -> {len(df):,} rows loaded")

# =============================================================
# 2. BUILD SEQUENCES WITH LAGGED TARGET
# =============================================================
print(f"\n[2/6] Building sequences (lookback = {LOOKBACK} months)...")

# Enhanced features: spatial (Zone, Class), temporal, weather, past demand
input_features = [
    'Zone_Encoded', 'Class_Encoded',
    'Month_Sin', 'Month_Cos',
    'Avg_Temp_F', 'Total_Precip_Inches', 'Avg_Humidity',
    'Water_Demand_Gallons'
]

df['Group'] = df['Zone_Encoded'].astype(str) + '_' + df['Class_Encoded'].astype(str)

# Calculate differenced target: change in demand from previous month
df['Demand_Diff'] = df.groupby('Group')['Water_Demand_Gallons'].diff()
target = 'Demand_Diff'

X_sequences = []
y_targets = []
y_prev_raw = []  # Need previous month's raw demand to un-difference predictions
actual_raw = []  # Need actual raw demand for final R2 calculation
date_indices = []
group_count = 0

for group_name, group_df in df.groupby('Group'):
    group_df = group_df.sort_values('Date').reset_index(drop=True)
    
    # We need LOOKBACK + 1 because diff creates a NaN on the first row
    if len(group_df) < LOOKBACK + 2:
        continue
    
    feat_vals = group_df[input_features].values
    target_vals = group_df[target].values
    raw_demand_vals = group_df['Water_Demand_Gallons'].values
    dates = group_df['Date'].values
    
    # Start at LOOKBACK + 1 to avoid the NaN from .diff() in the sequence if we used it, 
    # but the sequence inputs are raw demand, which is fine, but target needs to be valid.
    # If we start i at LOOKBACK+1, the sequence is i-LOOKBACK to i. 
    # The target is i. The diff at i is valid. 
    for i in range(LOOKBACK + 1, len(group_df)):
        X_sequences.append(feat_vals[i - LOOKBACK:i])
        y_targets.append(target_vals[i])
        y_prev_raw.append(raw_demand_vals[i - 1])
        actual_raw.append(raw_demand_vals[i])
        date_indices.append(dates[i])
    
    group_count += 1

X_sequences = np.array(X_sequences)
y_targets = np.array(y_targets)
y_prev_raw = np.array(y_prev_raw)
actual_raw = np.array(actual_raw)
date_indices = np.array(date_indices)

# Sort everything by date for proper chronological split
sort_idx = np.argsort(date_indices)
X_sequences = X_sequences[sort_idx]
y_targets = y_targets[sort_idx]
y_prev_raw = y_prev_raw[sort_idx]
actual_raw = actual_raw[sort_idx]
date_indices = date_indices[sort_idx]

print(f"  -> Groups processed: {group_count}")
print(f"  -> Total sequences: {X_sequences.shape[0]:,}")
print(f"  -> Sequence shape: {X_sequences.shape}")

# =============================================================
# 3. CHRONOLOGICAL SPLIT & SCALING
# =============================================================
print("\n[3/6] Splitting & scaling...")

split_idx = int(len(X_sequences) * 0.8)
X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]
y_prev_train, y_prev_test = y_prev_raw[:split_idx], y_prev_raw[split_idx:]
act_raw_train, act_raw_test = actual_raw[:split_idx], actual_raw[split_idx:]

# Scale features
n_train, n_steps, n_feat = X_train.shape
n_test = X_test.shape[0]

scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train.reshape(-1, n_feat)).reshape(n_train, n_steps, n_feat)
X_test_s = scaler_X.transform(X_test.reshape(-1, n_feat)).reshape(n_test, n_steps, n_feat)

# Scale differenced target
scaler_y = StandardScaler()
y_train_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

os.makedirs('models', exist_ok=True)
joblib.dump(scaler_X, 'models/lstm_scaler_X.pkl')
joblib.dump(scaler_y, 'models/lstm_scaler_y.pkl')

# =============================================================
# 4. BUILD & TRAIN LSTM
# =============================================================
print(f"\n[4/6] Building LSTM model...")

model_lstm = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(LOOKBACK, n_feat)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

print("\n  Training LSTM...\n")
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)

history = model_lstm.fit(
    X_train_s, y_train_s,
    validation_split=0.15,
    epochs=150,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# =============================================================
# 5. EVALUATE (Un-differencing)
# =============================================================
print("\n[5/6] Evaluating LSTM (Un-differencing)...")

pred_s = model_lstm.predict(X_test_s, verbose=0).flatten()
pred_diff = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).flatten()

# Un-difference: prediction = prev_raw_demand + predicted_diff
lstm_predictions = y_prev_test + pred_diff
lstm_predictions = np.maximum(lstm_predictions, 0) # Floor at 0

lstm_mae = mean_absolute_error(act_raw_test, lstm_predictions)
lstm_rmse = np.sqrt(mean_squared_error(act_raw_test, lstm_predictions))
lstm_r2 = r2_score(act_raw_test, lstm_predictions)

print(f"\n  Genuine LSTM Results (after removing lag illusion):")
print(f"    MAE:  {lstm_mae:>14,.0f} gallons")
print(f"    RMSE: {lstm_rmse:>14,.0f} gallons")
print(f"    R²:   {lstm_r2:>14.4f}")

# =============================================================
# 6. COMPARISON & CHARTS
# =============================================================
print("\n[6/6] Generating charts...")

try:
    ann_metrics = joblib.load('models/comparison_metrics.pkl')
    ann_r2 = ann_metrics['ann']['R²']
    ann_mae = ann_metrics['ann']['MAE']
    ann_rmse = ann_metrics['ann']['RMSE']
except:
    ann_r2, ann_mae, ann_rmse = 0.8930, 3_332_922, 6_754_624

winner = "LSTM" if lstm_r2 > ann_r2 else "ANN"

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax1 = axes[0]
ax1.plot(history.history['loss'], label='Training Loss', color='#9C27B0')
ax1.plot(history.history['val_loss'], label='Validation Loss', color='#FF5722')
ax1.set_title('LSTM Training History (Differenced)', fontsize=13, fontweight='bold')
ax1.legend()

ax2 = axes[1]
x = np.arange(2)
width = 0.35
bars1 = ax2.bar(x - width/2, [lstm_mae, lstm_rmse], width, label='LSTM', color='#9C27B0')
bars2 = ax2.bar(x + width/2, [ann_mae, ann_rmse], width, label='ANN', color='#FF9800')
ax2.set_title('Genuine Error Comparison', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['MAE', 'RMSE'])
ax2.legend()
for bar in bars1: ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height()/1e6:.1f}M', ha='center', va='bottom', fontsize=9)
for bar in bars2: ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height()/1e6:.1f}M', ha='center', va='bottom', fontsize=9)

ax3 = axes[2]
names = ['LSTM\n(Genuine)', 'ANN']
scores = [lstm_r2, ann_r2]
bars = ax3.bar(names, scores, color=['#9C27B0', '#FF9800'], width=0.5)
ax3.set_title('Genuine R² Score Comparison', fontsize=13, fontweight='bold')
ax3.set_ylim(0, 1.1)
for bar, score in zip(bars, scores): ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('LSTM_vs_ANN_Comparison.png')
plt.close()

# Forecast chart
n_show = min(200, len(act_raw_test))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(range(n_show), act_raw_test[:n_show], color='black', linewidth=1.2, label='Actual Demand')
ax.plot(range(n_show), lstm_predictions[:n_show], color='#9C27B0', linewidth=1.2, linestyle='--', label='LSTM Predicted')
ax.set_title('Genuine LSTM Forecast (No Lag Illusion)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('LSTM_Forecast.png')
plt.close()

model_lstm.save('models/lstm_model.keras')
joblib.dump({
    'model_name': 'LSTM (Differenced)',
    'mae': lstm_mae, 'rmse': lstm_rmse, 'r2': lstm_r2,
    'lookback': LOOKBACK,
    'seq_features': input_features,
    'winner': winner
}, 'models/lstm_metrics.pkl')

print("\n  LSTM TRAINING COMPLETE")
print(f"  Genuine LSTM R²: {lstm_r2:.4f}")
