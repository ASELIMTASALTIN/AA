import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf

st.set_page_config(layout="wide")
st.title("📊 Machine Learning for Dopamine Detection")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    feature_options = list(df.columns)
    target = st.selectbox("Select target column (e.g., Dopamine_Concentration):", feature_options)
    feature_options.remove(target)
    selected_features = st.multiselect("Select feature columns to use:", feature_options, default=feature_options[:3])

    if st.button("Run Machine Learning"):
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

        # ANN
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

        # Display Results
        results_df = pd.DataFrame(results).T
        st.subheader("📋 Performance Metrics")
        st.dataframe(results_df)

        residuals_df = pd.DataFrame(residual_data)
        residuals_df['Error'] = residuals_df['True'] - residuals_df['Predicted']

        # 2D Residuals Plot
        st.subheader("📈 Residual Plot (True vs Predicted)")
        fig2d, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=residuals_df, x='True', y='Predicted', hue='Model', s=80, ax=ax)
        ax.plot([min(y), max(y)], [min(y), max(y)], 'k--')
        st.pyplot(fig2d)

        # 3D Interactive Plot
        st.subheader("🌐 3D Plot (True vs Predicted vs Error)")
        fig3d = px.scatter_3d(
            residuals_df,
            x='True',
            y='Predicted',
            z='Error',
            color='Model',
            size_max=10
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # CSV Download
        st.download_button("📥 Download Results CSV", results_df.to_csv().encode(), file_name="model_results.csv")
        st.download_button("📥 Download Residuals CSV", residuals_df.to_csv(index=False).encode(), file_name="model_residuals.csv")
