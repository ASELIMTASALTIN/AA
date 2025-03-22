...import streamlit as st
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

# Page settings
st.set_page_config(layout="wide", page_title="Ahmet Selim Taşaltın's Comparative Engine")
st.markdown("<h1 style='text-align: center;'>🔬 Ahmet Selim Taşaltın's Comparative Engine</h1>", unsafe_allow_html=True)

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Feature selection
    feature_options = list(df.columns)
    target = st.selectbox("🎯 Select your target column", feature_options)
    feature_options.remove(target)
    selected_features = st.multiselect("🧪 Select feature columns", feature_options, default=feature_options[:2])

    if st.button("🚀 Run Model Comparison"):
        X = df[selected_features].values
        y = df[target].values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=20),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.3, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=10, learning_rate=0.1),
            'SVR': SVR(kernel='rbf', C=100, epsilon=0.1)
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

        # ANN Model
        def build_ann_model(input_dim):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(32, activation='relu'),
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

        # Create dataframe for metrics and residuals
        results_df = pd.DataFrame(results).T
        residuals_df = pd.DataFrame(residual_data)
        residuals_df['Error'] = residuals_df['True'] - residuals_df['Predicted']

        # Main visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📈 Residuals: True vs Predicted")
            fig2d, ax = plt.subplots()
            sns.scatterplot(data=residuals_df, x='True', y='Predicted', hue='Model', s=80, ax=ax)
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
                    line=dict(color=model_colors[model]),
                    marker=dict(size=4)
                ))
            fig3d.update_layout(
                scene=dict(xaxis_title='True', yaxis_title='Predicted', zaxis_title='Error'),
                height=500
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # Add ANN and SVR predictions to dataframe for individual plotting
        # Train final ANN model on all data for visualization
        final_ann_model = build_ann_model(X_scaled.shape[1])
        final_ann_model.fit(X_scaled, y, epochs=150, verbose=0)
        df['ANN_Predicted'] = final_ann_model.predict(X_scaled).flatten()

        # Train SVR on all data for plotting
        final_svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
        df['SVR_Predicted'] = final_svr_model.fit(X_scaled, y).predict(X_scaled)


        st.markdown("---")
        st.markdown("## 📊 Comparative Graphs: Experimental vs ML Predictions")
        x_feature = st.selectbox("🧭 Select x-axis", selected_features)
        samples = df['Sample'].unique() if 'Sample' in df.columns else ['All']

        for i in range(0, len(samples), 2):
            row = st.columns(2)
            for j in range(2):
                if i + j < len(samples):
                    sample = samples[i + j]
                    sample_df = df[df['Sample'] == sample] if sample != 'All' else df

                    fig, ax = plt.subplots()
                    ax.plot(sample_df[x_feature], sample_df[target], label='Experimental Data', color='red')
                    ax.plot(sample_df[x_feature], sample_df['ANN_Predicted'], label='Predicted Data (ANN)', linestyle='dotted', color='blue')
                    ax.plot(sample_df[x_feature], sample_df['SVR_Predicted'], label='Predicted Data (SVR)', linestyle='dotted', color='black')
                    if (sample_df[target] > 0).all():
                        ax.set_yscale("log")
                    ax.set_title(f"{target} vs {x_feature} — {sample}")
                    ax.set_xlabel(x_feature)
                    ax.set_ylabel(target)
                    ax.grid(True)
                    ax.legend()
                    row[j].pyplot(fig)

        # Bottom row with additional insights
        st.markdown("----")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### 📊 Metrics Comparison")
            fig_bar, ax = plt.subplots()
            results_df[['MAE', 'RMSE', 'R2']].plot(kind='bar', ax=ax)
            ax.set_title("Performance Metrics per Model")
            ax.set_ylabel("Score")
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig_bar)

        with col4:
            st.markdown("### 📉 Error Distributions")
            fig_dist, ax = plt.subplots()
            for model in residuals_df['Model'].unique():
                sns.kdeplot(data=residuals_df[residuals_df['Model'] == model], x='Error', label=model, ax=ax, fill=True)
            ax.set_xlabel("Error")
            ax.set_title("Residual Distributions")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig_dist)

        # Results Table and Downloads
        st.markdown("---")
        st.markdown("### 📋 Final Results")
        st.dataframe(results_df.style.format("{:.4f}"))
        st.download_button("📥 Download Metrics", results_df.to_csv().encode(), file_name="model_metrics.csv")
        st.download_button("📥 Download Residuals", residuals_df.to_csv(index=False).encode(), file_name="residuals.csv")
