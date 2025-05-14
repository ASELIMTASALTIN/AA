import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

# Streamlit setup
st.set_page_config(layout="wide")
st.title("🔬 Curve-wise Concentration Prediction")

uploaded_file = st.file_uploader("📥 Upload your CSV with Electrochemical Features", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("### 📋 Raw Data")
    st.dataframe(df)

    # Validate necessary columns
    if "Concentration" not in df.columns or "Curve" not in df.columns:
        st.error("CSV must include 'Curve' and 'Concentration' columns.")
    else:
        # Feature matrix and target
        X = df.drop(columns=["Curve", "Concentration"]).values
        y = df["Concentration"].values

        models = {
            "Linear Regression": LinearRegression(),
            "SVR": SVR(kernel='rbf', C=100, epsilon=0.01)
        }

        results = {}
        loo = LeaveOneOut()

        # Evaluate classical models
        for name, model in models.items():
            y_true, y_pred = [], []
            for train_idx, test_idx in loo.split(X):
                model.fit(X[train_idx], y[train_idx])
                pred = model.predict(X[test_idx])
                y_true.append(y[test_idx][0])
                y_pred.append(pred[0])
                rmse = mean_squared_error(y_true, y_pred) ** 0.5

            r2 = r2_score(y_true, y_pred)
            results[name] = {"RMSE": rmse, "R2": r2}

        # ANN model
        def build_ann(input_dim):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='linear')
            ])
            model.compile(optimizer='adam', loss='mse')
            return model

        y_true_ann, y_pred_ann = [], []

        for train_idx, test_idx in loo.split(X):
            ann_model = build_ann(X.shape[1])
            ann_model.fit(X[train_idx], y[train_idx], epochs=200, verbose=0)
            pred = ann_model.predict(X[test_idx], verbose=0)
            y_true_ann.append(float(y[test_idx][0]))
            y_pred_ann.append(float(pred[0][0]))

        # Compute ANN metrics
        if y_true_ann and y_pred_ann:
            rmse_ann = mean_squared_error(y_true_ann, y_pred_ann) ** 0.5
            r2_ann = r2_score(y_true_ann, y_pred_ann)
            results["ANN"] = {"RMSE": rmse_ann, "R2": r2_ann}

        # Display results
        st.markdown("## 📊 Model Performance Metrics")
        for model, metric in results.items():
            st.markdown(f"**{model}** — RMSE: `{metric['RMSE']:.5f}`, R²: `{metric['R2']:.5f}`")
