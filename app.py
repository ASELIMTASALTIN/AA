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
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Set wide layout and custom title
st.set_page_config(layout="wide", page_title="Ahmet Selim Taşaltın's Comparative Engine")
st.markdown("<h1 style='text-align: center;'>📊 Ahmet Selim Taşaltın's Comparative Engine</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload your CSV file with features and target", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    feature_options = list(df.columns)
    target = st.selectbox("🎯 Select your target column", feature_options)
    feature_options.remove(target)
    selected_features = st.multiselect("🧪 Select feature columns", feature_options, default=feature_options[:3])

    svr_c_value = st.slider("🔧 SVR: Select C parameter", min_value=0.01, max_value=100.0, value=1.0, step=0.1)

    if st.button("🚀 Run Comparative Engine"):
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
            'ANN': 'red',
            'SVR': 'black'
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

        def build_ann_model(input_dim):
            model = Sequential([
                Input(shape=(input_dim,)),
                Dense(32, activation='relu'),
                Dense(128, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='linear')
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

        # SVR Model (with adjustable C)
        svr_model = SVR(kernel='rbf', C=svr_c_value)
        y_true_svr, y_pred_svr = [], []
        for train_idx, test_idx in LeaveOneOut().split(X_scaled):
            svr_model.fit(X_scaled[train_idx], y[train_idx])
            pred = svr_model.predict(X_scaled[test_idx])
            y_true_svr.append(y[test_idx][0])
            y_pred_svr.append(pred[0])
            residual_data.append({'Model': 'SVR', 'True': y[test_idx][0], 'Predicted': pred[0]})
        results['SVR'] = {
            'MAE': mean_absolute_error(y_true_svr, y_pred_svr),
            'RMSE': mean_squared_error(y_true_svr, y_pred_svr) ** 0.5,
            'R2': r2_score(y_true_svr, y_pred_svr)
        }

        results_df = pd.DataFrame(results).T
        residuals_df = pd.DataFrame(residual_data)
        residuals_df['Error'] = residuals_df['True'] - residuals_df['Predicted']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Residuals: True vs Predicted")
            fig2d, ax = plt.subplots(figsize=(6, 5))
            sns.scatterplot(data=residuals_df, x='True', y='Predicted', hue='Model', s=90, ax=ax)
            ax.plot([min(y), max(y)], [min(y), max(y)], 'k--')
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.grid(True)
            st.pyplot(fig2d)

        with col2:
            st.markdown("### 🌐 3D: True vs Predicted vs Error")
            fig3d = go.Figure()
            for model in residuals_df['Model'].unique():
                model_data = residuals_df[residuals_df['Model'] == model]
                fig3d.add_trace(go.Scatter3d(
                    x=model_data['True'],
                    y=model_data['Predicted'],
                    z=model_data['Error'],
                    mode='lines+markers',
                    name=model,
                    line=dict(color=model_colors.get(model, 'gray')),
                    marker=dict(size=5)
                ))
            fig3d.update_layout(
                scene=dict(xaxis_title='True', yaxis_title='Predicted', zaxis_title='Error'),
                margin=dict(l=0, r=0, t=40, b=0),
                height=500
            )
            st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📋 Performance Metrics")
        st.dataframe(results_df.style.format("{:.4f}"))

        # Download buttons
        st.download_button("📥 Download Results CSV", results_df.to_csv().encode(), file_name="model_results.csv")
        st.download_button("📥 Download Residuals CSV", residuals_df.to_csv(index=False).encode(), file_name="model_residuals.csv")
