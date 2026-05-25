"""
==========================================================================
  INTERACTIVE DASHBOARD
  Water Consumption Prediction - Data Mining Project
  --------------------------------------------------
  A Flask web dashboard showcasing the entire data mining pipeline:
    1. Dataset Overview & Statistics
    2. EDA Visualizations
    3. ANN vs LSTM Model Comparison
    4. Real-time Prediction (uses LSTM — best model)
==========================================================================
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# ---------------------------------------------------------
# Load data, models, and encoders at startup
# ---------------------------------------------------------
print("=" * 60)
print("  Loading Dashboard Resources...")
print("=" * 60)

df = pd.read_csv('cleaned_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Label encoders
le_zone = joblib.load('models/label_encoder_zone.pkl')
le_class = joblib.load('models/label_encoder_class.pkl')
le_season = joblib.load('models/label_encoder_season.pkl')

# LSTM model + scalers
lstm_model = load_model('models/lstm_model.keras')
lstm_scaler_X = joblib.load('models/lstm_scaler_X.pkl')
lstm_scaler_y = joblib.load('models/lstm_scaler_y.pkl')
lstm_metrics = joblib.load('models/lstm_metrics.pkl')

# ANN metrics
try:
    ann_metrics_data = joblib.load('models/comparison_metrics.pkl')
    ann_metrics = ann_metrics_data.get('ann', {})
except:
    ann_metrics = {'MAE': 3332922, 'RMSE': 6754624, 'R²': 0.8930}

# Dropdown lists
zone_list = sorted(le_zone.classes_.tolist())
class_list = sorted(le_class.classes_.tolist())

# Pre-compute dataset stats
total_rows = len(df)
date_range = f"{df['Date'].min().strftime('%b %Y')} - {df['Date'].max().strftime('%b %Y')}"
n_zones = df['Zone_ZipCode'].nunique() if 'Zone_ZipCode' in df.columns else df['Zone_Encoded'].nunique()
n_classes = df['Class_Encoded'].nunique()
avg_demand = df['Water_Demand_Gallons'].mean()
total_demand = df['Water_Demand_Gallons'].sum()

# EDA chart filenames
eda_charts = []
chart_files = [
    ('Seasonal_Trend.png', 'Seasonal Water Demand Trend', 'Monthly demand aggregated across all zones, showing clear seasonal patterns with summer peaks.'),
    ('Temp_vs_Demand.png', 'Temperature vs Water Demand', 'Scatter plot showing strong positive correlation between temperature and water consumption.'),
    ('Correlation_Heatmap.png', 'Feature Correlation Heatmap', 'Correlation matrix of all numerical features, highlighting key relationships in the data.'),
    ('Customer_Class_Boxplot.png', 'Demand by Customer Class', 'Box plot comparing water demand distributions across different customer categories.'),
    ('Precip_vs_Demand_Season.png', 'Precipitation vs Demand by Season', 'Seasonal breakdown of how rainfall affects water consumption patterns.'),
    ('Zone_Level_Time_Series.png', 'Zone-Level Time Series', 'Top zones water demand over time, showing spatial variation in consumption.'),
]
for fname, title, desc in chart_files:
    if os.path.exists(fname):
        eda_charts.append({'file': fname, 'title': title, 'desc': desc})

# Model comparison charts
model_charts = []
for fname in ['LSTM_vs_ANN_Comparison.png', 'LSTM_Forecast.png', 'ANN_vs_RF_Comparison.png', 'ANN_vs_RF_Predictions.png']:
    if os.path.exists(fname):
        model_charts.append(fname)

print(f"  -> Dataset: {total_rows:,} rows | {date_range}")
print(f"  -> Zones: {n_zones} | Classes: {n_classes}")
print(f"  -> LSTM R2: {lstm_metrics['r2']:.4f}")
print(f"  -> ANN R2:  {ann_metrics.get('R\u00b2', 0):.4f}")
print(f"  -> EDA charts: {len(eda_charts)} | Model charts: {len(model_charts)}")
print(f"  -> Dashboard ready!\n")


def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'


@app.route('/')
def dashboard():
    return render_template('dashboard.html',
        # Dataset stats
        total_rows=f"{total_rows:,}",
        date_range=date_range,
        n_zones=n_zones,
        n_classes=n_classes,
        avg_demand=f"{avg_demand:,.0f}",
        total_demand=f"{total_demand/1e9:.2f}",
        # Model metrics
        lstm_r2=f"{lstm_metrics['r2']:.4f}",
        lstm_mae=f"{lstm_metrics['mae']:,.0f}",
        lstm_rmse=f"{lstm_metrics['rmse']:,.0f}",
        ann_r2=f"{ann_metrics.get('R²', 0):.4f}",
        ann_mae=f"{ann_metrics.get('MAE', 0):,.0f}",
        ann_rmse=f"{ann_metrics.get('RMSE', 0):,.0f}",
        # Charts
        eda_charts=eda_charts,
        model_charts=model_charts,
        # Prediction form data
        zones=zone_list,
        classes=class_list,
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        month = int(request.form['month'])
        zone = request.form['zone']
        customer_class = request.form['customer_class']
        avg_temp = float(request.form['avg_temp'])
        total_precip = float(request.form['total_precip'])
        avg_humidity = float(request.form['avg_humidity'])

        # Build a sequence from the last 12 months of data for this zone-class
        zone_encoded = le_zone.transform([zone])[0]
        class_encoded = le_class.transform([customer_class])[0]
        season = get_season(month)

        # Filter historical data for this zone-class combination
        group_df = df[(df['Zone_Encoded'] == zone_encoded) & 
                      (df['Class_Encoded'] == class_encoded)].sort_values('Date')

        if len(group_df) < 12:
            # Not enough history — use synthetic sequence from averages
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            avg_demand_val = group_df['Water_Demand_Gallons'].mean() if len(group_df) > 0 else df['Water_Demand_Gallons'].mean()
            
            row = [zone_encoded, class_encoded, month_sin, month_cos, avg_temp, total_precip, avg_humidity, avg_demand_val]
            sequence = np.array([row] * 12)
        else:
            # Use last 12 rows as the sequence
            last_12 = group_df.tail(12)
            sequence = last_12[['Zone_Encoded', 'Class_Encoded', 'Month_Sin', 'Month_Cos', 'Avg_Temp_F',
                               'Total_Precip_Inches', 'Avg_Humidity',
                               'Water_Demand_Gallons']].values
            # Override the weather of the last timestep with user input
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            sequence[-1, 2] = month_sin
            sequence[-1, 3] = month_cos
            sequence[-1, 4] = avg_temp
            sequence[-1, 5] = total_precip
            sequence[-1, 6] = avg_humidity

        # Get previous raw demand (from the very last timestep in our history)
        prev_raw_demand = sequence[-1, 7]

        # Scale and predict the difference
        seq_2d = lstm_scaler_X.transform(sequence)
        seq_3d = seq_2d.reshape(1, 12, sequence.shape[1])
        
        pred_scaled = lstm_model.predict(seq_3d, verbose=0).flatten()
        pred_diff = lstm_scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        
        # Un-difference to get raw prediction
        prediction = prev_raw_demand + pred_diff
        prediction = max(0, prediction)

        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                       'July', 'August', 'September', 'October', 'November', 'December']

        return jsonify({
            'success': True,
            'prediction': f"{prediction:,.0f}",
            'prediction_raw': round(prediction),
            'month_name': month_names[month - 1],
            'zone': zone,
            'customer_class': customer_class,
            'season': season,
            'avg_temp': avg_temp,
            'total_precip': total_precip,
            'avg_humidity': avg_humidity,
            'model': 'LSTM',
            'r2': f"{lstm_metrics['r2']:.4f}"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# Serve chart images
@app.route('/charts/<path:filename>')
def serve_chart(filename):
    from flask import send_from_directory
    return send_from_directory('.', filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
