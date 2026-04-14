"""
==========================================================================
  EXPLORATORY DATA ANALYSIS (EDA)
  Water Consumption Prediction - Data Mining Project
  --------------------------------------------------
  Generates 6 publication-quality visualizations from the cleaned dataset.
==========================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set professional visual style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'

print("=" * 60)
print("  STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Load cleaned data
print("\nLoading cleaned dataset...")
df = pd.read_csv('cleaned_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
print(f"  → {len(df):,} rows loaded")

# ---------------------------------------------------------
# VISUALIZATION 1: Average Water Demand by Month (Seasonal)
# ---------------------------------------------------------
print("\n[1/6] Generating Seasonal Trend chart...")
fig, ax = plt.subplots(figsize=(10, 5))
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_avg = df.groupby('Month')['Water_Demand_Gallons'].mean().reset_index()
colors = ['#2196F3' if m in [6,7,8] else '#90CAF9' for m in monthly_avg['Month']]
bars = ax.bar(monthly_avg['Month'], monthly_avg['Water_Demand_Gallons'], color=colors, edgecolor='white')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_names)
ax.set_title('Average Water Demand by Month', fontsize=15, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Avg Water Demand (Gallons)')
ax.ticklabel_format(style='plain', axis='y')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height/1e6:.1f}M', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('Seasonal_Trend.png')
plt.close()
print("  → Saved: Seasonal_Trend.png")

# ---------------------------------------------------------
# VISUALIZATION 2: Temperature vs Water Demand
# ---------------------------------------------------------
print("[2/6] Generating Temperature vs Demand scatter plot...")
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['Avg_Temp_F'], df['Water_Demand_Gallons'],
                     c=df['Month'], cmap='RdYlBu_r', alpha=0.3, s=10, edgecolors='none')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Month')

# Add trend line
z = np.polyfit(df['Avg_Temp_F'], df['Water_Demand_Gallons'], 1)
p = np.poly1d(z)
temp_range = np.linspace(df['Avg_Temp_F'].min(), df['Avg_Temp_F'].max(), 100)
ax.plot(temp_range, p(temp_range), color='red', linewidth=2, label=f'Trend Line')

ax.set_title('Impact of Temperature on Water Consumption', fontsize=15, fontweight='bold')
ax.set_xlabel('Average Temperature (°F)')
ax.set_ylabel('Water Demand (Gallons)')
ax.legend()
ax.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.savefig('Temp_vs_Demand.png')
plt.close()
print("  → Saved: Temp_vs_Demand.png")

# ---------------------------------------------------------
# VISUALIZATION 3: Top 5 Zones — Time Series
# ---------------------------------------------------------
print("[3/6] Generating Zone-level Time Series...")
top_5_zones = df.groupby('Zone_ZipCode')['Water_Demand_Gallons'].sum().nlargest(5).index
zone_df = df[df['Zone_ZipCode'].isin(top_5_zones)]

fig, ax = plt.subplots(figsize=(14, 6))
for zone in top_5_zones:
    zdata = zone_df[zone_df['Zone_ZipCode'] == zone].groupby('Date')['Water_Demand_Gallons'].sum().reset_index()
    ax.plot(zdata['Date'], zdata['Water_Demand_Gallons'], marker='o', markersize=3, label=f'Zone {zone}', linewidth=1.5)

ax.set_title('Monthly Water Demand — Top 5 Zones', fontsize=15, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Total Water Demand (Gallons)')
ax.legend(title='Zip Code')
ax.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.savefig('Zone_Level_Time_Series.png')
plt.close()
print("  → Saved: Zone_Level_Time_Series.png")

# ---------------------------------------------------------
# VISUALIZATION 4: Correlation Heatmap
# ---------------------------------------------------------
print("[4/6] Generating Correlation Heatmap...")
corr_cols = ['Water_Demand_Gallons', 'Avg_Temp_F', 'Total_Precip_Inches',
             'Avg_Humidity', 'Month', 'Month_Sin', 'Month_Cos']
corr_matrix = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            vmin=-1, vmax=1, mask=mask, linewidths=0.5,
            square=True, ax=ax,
            annot_kws={"size": 10})
ax.set_title('Feature Correlation Matrix', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('Correlation_Heatmap.png')
plt.close()
print("  → Saved: Correlation_Heatmap.png")

# ---------------------------------------------------------
# VISUALIZATION 5: Demand Distribution by Customer Class
# ---------------------------------------------------------
print("[5/6] Generating Customer Class Box Plot...")
fig, ax = plt.subplots(figsize=(10, 6))
class_order = df.groupby('Customer Class')['Water_Demand_Gallons'].median().sort_values(ascending=False).index
palette = {'Residential': '#1976D2', 'Multi-Family': '#388E3C',
           'Irrigation - Multi-Family': '#F57C00', 'Irrigation - Residential': '#D32F2F'}

sns.boxplot(data=df, x='Customer Class', y='Water_Demand_Gallons',
            order=class_order, palette=palette, fliersize=1, ax=ax)
ax.set_title('Water Demand Distribution by Customer Class', fontsize=15, fontweight='bold')
ax.set_xlabel('Customer Class')
ax.set_ylabel('Water Demand (Gallons)')
ax.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('Customer_Class_Boxplot.png')
plt.close()
print("  → Saved: Customer_Class_Boxplot.png")

# ---------------------------------------------------------
# VISUALIZATION 6: Precipitation vs Demand by Season
# ---------------------------------------------------------
print("[6/6] Generating Precipitation vs Demand by Season...")
fig, ax = plt.subplots(figsize=(10, 6))
season_colors = {'Winter': '#2196F3', 'Spring': '#4CAF50', 'Summer': '#FF9800', 'Fall': '#9C27B0'}
for season, color in season_colors.items():
    mask = df['Season'] == season
    ax.scatter(df.loc[mask, 'Total_Precip_Inches'],
               df.loc[mask, 'Water_Demand_Gallons'],
               c=color, label=season, alpha=0.3, s=10, edgecolors='none')

ax.set_title('Precipitation vs Water Demand (by Season)', fontsize=15, fontweight='bold')
ax.set_xlabel('Total Precipitation (Inches)')
ax.set_ylabel('Water Demand (Gallons)')
ax.legend(title='Season', markerscale=3)
ax.ticklabel_format(style='plain', axis='y')
plt.tight_layout()
plt.savefig('Precip_vs_Demand_Season.png')
plt.close()
print("  → Saved: Precip_vs_Demand_Season.png")

# ---------------------------------------------------------
print("\n" + "=" * 60)
print("  EDA COMPLETE — 6 charts saved as PNG files")
print("=" * 60)