import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf

# Load the dataset
df = pd.read_csv("dopamine_data.csv")

features = ['Charge', 'PeakCurrent', 'Skewness']
X = df[features].values
y = df['Dopamine_Concentration'].values

# Scale input features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=20),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.3, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=5, learning_rate=0.1)
}

results = {}
residual_data = []

# Leave-One-Out Cross-Validation for traditional models
for name, model in models.items():
    y_true, y_pred = [], []
    for train_idx, test_idx in LeaveOneOut().split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        pred = model.predict(X_scaled[test_idx])
        y_true.append(y[test_idx][0])
        y_pred.append(pred[0])
        residual_data.append({'Model': name, 'True': y[test_idx][0], 'Predicted': pred[0]})
    results[name] = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': mean_squared_error(y_true, y_pred) ** 0.5,
        'R2': r2_score(y_true, y_pred)
    }

# ANN model
def build_ann_model(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
    return model

y_true_ann, y_pred_ann = [], []
for train_idx, test_idx in LeaveOneOut().split(X_scaled):
    ann = build_ann_model(X_scaled.shape[1])
    ann.fit(X_scaled[train_idx], y[train_idx], epochs=150, verbose=0)
    pred = ann.predict(X_scaled[test_idx])
    y_true_ann.append(y[test_idx][0])
    y_pred_ann.append(pred[0][0])
    residual_data.append({'Model': 'ANN', 'True': y[test_idx][0], 'Predicted': pred[0][0]})
results['ANN'] = {
    'MAE': mean_absolute_error(y_true_ann, y_pred_ann),
    'RMSE': mean_squared_error(y_true_ann, y_pred_ann) ** 0.5,
    'R2': r2_score(y_true_ann, y_pred_ann)
}

# Save results to CSV
results_df = pd.DataFrame(results).T
results_df.to_csv("model_results.csv")

# Save residuals to CSV
residuals_df = pd.DataFrame(residual_data)
residuals_df.to_csv("model_residuals.csv", index=False)

# --- Plot 1: Bar Plot of Performance Metrics ---
plt.figure(figsize=(10, 6))
results_df[['MAE', 'RMSE', 'R2']].plot(kind='bar')
plt.title("Model Performance Metrics")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig("performance_metrics.png")
plt.show()
# --- Plot 2: Residuals Plot ---
plt.figure(figsize=(10, 6))
sns.scatterplot(data=residuals_df, x='True', y='Predicted', hue='Model', s=100)
plt.plot([min(y), max(y)], [min(y), max(y)], 'k--', label="Ideal")
plt.title("True vs Predicted Dopamine Concentrations")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("residuals_plot.png")
plt.show()
# --- 3D Plot: Charge vs True vs Predicted ---
# Color mapping for each model (used in both 3D plots)
model_colors = {
    'Linear Regression': 'blue',
    'Random Forest': 'green',
    'XGBoost': 'orange',
    'LightGBM': 'purple',
    'ANN': 'red'
}

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Use 'Charge' as the X-axis feature
for model in residuals_df['Model'].unique():
    model_data = residuals_df[residuals_df['Model'] == model].copy()
    # Join feature from original X (remember: scaled, so reverse scale or use original Charge)
    model_data['Charge'] = np.tile(df['Charge'].values, int(len(model_data)/len(df)))
    ax.scatter(
        model_data['Charge'],
        model_data['True'],
        model_data['Predicted'],
        c=model_colors[model],
        label=model,
        s=60
    )

ax.set_xlabel('Charge')
ax.set_ylabel('True Concentration')
ax.set_zlabel('Predicted Concentration')
ax.set_title('3D Plot: Charge vs True vs Predicted')
ax.legend()
plt.tight_layout()
plt.savefig("3D_charge_true_pred.png")
plt.show()
