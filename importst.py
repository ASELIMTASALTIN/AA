import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

st.title("🔬 Concentration Prediction — Curve-wise Evaluation")

uploaded_file = st.file_uploader("Upload your CSV with Electrochemical Features", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.dataframe(df)

    X = df.drop(columns=["Curve", "Concentration"]).values
    y = df["Concentration"].values

    models = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(kernel='rbf', C=100, epsilon=0.01)
    }

    results = {}

    loo = LeaveOneOut()

    for name, model in models.items():
        y_true, y_pred = [], []
        for train_idx, test_idx in loo.split(X):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            y_true.append(y[test_idx][0])
            y_pred.append(pred[0])
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
            r2 = r2_score(y_true, y_pred)

        r2 = r2_score(y_true, y_pred)
        results[name] = {"RMSE": rmse, "R2": r2}

    # ANN Model
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
        pred = ann_model.predict(X[test_idx])
        y_true_ann.append(y[test_idx][0])
        y_pred_ann.append(pred[0][0])

        rmse_ann = mean_squared_error(y_true_ann, y_pred_ann) ** 0.5
        r2_ann = r2_score(y_true_ann, y_pred_ann)

    r2_ann = r2_score(y_true_ann, y_pred_ann)
    results["ANN"] = {"RMSE": rmse_ann, "R2": r2_ann}

    st.markdown("## 📊 Model Performance")
    for model, metrics in results.items():
        st.markdown(f"**{model}** — RMSE: `{metrics['RMSE']:.5f}` , R²: `{metrics['R2']:.5f}`")
