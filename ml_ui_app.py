import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf

# Set wide layout and custom title
st.set_page_config(layout="wide", page_title="Dopamine ML Model App")
st.markdown("<h1 style='text-align: center;'>🧠 Machine Learning for Dopamine Detection</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your CSV file with features and target", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    feature_options = list(df.columns)
    target = st.selectbox("🎯 Select your target column (e.g., Dopamine_Concentration)", feature_options)
    feature_options.remove(target)
    selected_features = st.multiselect("🧪 Select feature columns", feature_options, default=feature_options[:3])

    if st.button("🚀 Run Machine Learning Pipeline"):
        X = df[selected_features].values
        y = df[target].values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=20),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.3, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=5, learning_rate=0.1)
        }

        model_colors = {
            'Linear Regression': 'blue',
            'Random Forest': 'green',
            'XGBoost': 'orange',
            'LightGBM': 'purple',
            'ANN': 'red'
        }

        results = {}
        residual_data = []

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

        # ANN Model
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

        results_df = pd.DataFrame(results).T
        residuals_df = pd.DataFrame(residual_data)
        residuals_df['Error'] = residuals_df['True'] - residuals_df['Predicted']

        # Layout for side-by-side plots
        col1, col2 = st.columns([1, 1])

        # --- LEFT: 2D Residuals Plot ---
        with col1:
            st.markdown("### 📈 Residuals: True vs Predicted")
            fig2d, ax = plt.subplots(figsize=(6, 5))
            sns.scatterplot(data=residuals_df, x='True', y='Predicted', hue='Model', s=90, ax=ax)
            ax.plot([min(y), max(y)], [min(y), max(y)], 'k--')
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.grid(True)
            st.pyplot(fig2d)

        # --- RIGHT: 3D Plot with Lines ---
        with col2:
            st.markdown("### 🌐 3D: True vs Predicted vs Error (Connected by Model)")
            fig3d = go.Figure()

            for model in residuals_df['Model'].unique():
                model_data = residuals_df[residuals_df['Model'] == model]
                fig3d.add_trace(go.Scatter3d(
                    x=model_data['True'],
                    y=model_data['Predicted'],
                    z=model_data['Error'],
                    mode='lines+markers',
                    name=model,
                    line=dict(color=model_colors[model]),
                    marker=dict(size=5)
                ))

            fig3d.update_layout(
                scene=dict(
                    xaxis_title='True',
                    yaxis_title='Predicted',
                    zaxis_title='Error',
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                height=500
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # Create another row of side-by-side charts
        st.markdown("----")
        col3, col4 = st.columns([1, 1])

        # --- LEFT: Bar Chart of Performance Metrics ---
        with col3:
          st.markdown("### 📊 Model Performance Metrics (Bar Chart)")
          fig_bar, ax = plt.subplots(figsize=(6, 5))
          results_df[['MAE', 'RMSE', 'R2']].plot(kind='bar', ax=ax)
          ax.set_ylabel("Score")
          ax.set_title("MAE / RMSE / R² per Model")
          ax.grid(True)
          plt.xticks(rotation=45)
          st.pyplot(fig_bar)

  # --- RIGHT: Error Distribution Plot ---
        with col4:
          st.markdown("### 📉 Error Distribution per Model")
          fig_dist, ax = plt.subplots(figsize=(6, 5))
          for model in residuals_df['Model'].unique():
           sns.kdeplot(
            data=residuals_df[residuals_df['Model'] == model],
            x='Error',
            label=model,
            ax=ax,
            fill=True,
            alpha=0.4,
            linewidth=2
        )
        ax.set_xlabel("Error (True - Predicted)")
        ax.set_title("Residual Distributions")
        ax.grid(True)
        ax.legend()


        st.pyplot(fig_dist)
        st.markdown("---")
        st.markdown("### 📋 Performance Metrics")
        st.dataframe(results_df.style.format("{:.4f}"))

        # Download buttons
        st.download_button("📥 Download Results CSV", results_df.to_csv().encode(), file_name="model_results.csv")
        st.download_button("📥 Download Residuals CSV", residuals_df.to_csv(index=False).encode(), file_name="model_residuals.csv")
