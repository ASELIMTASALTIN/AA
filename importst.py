import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(layout="wide", page_title="Electrochemical Concentration Predictor")
st.title("🔬 Predict Concentration from Electrochemical Features")

uploaded_file = st.file_uploader("📂 Upload Extracted Features CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.dataframe(df.head())

    feature_cols = ['PeakCurrent', 'Charge_Q', 'Skewness', 'PeakSecondDerivative', 'Impedance', 'Conductance']
    target_col = 'Concentration'

    if not all(col in df.columns for col in feature_cols + [target_col]):
        st.error("❌ Required columns not found. Ensure your CSV includes: " + ", ".join(feature_cols + [target_col]))
    else:
        X = df[feature_cols].values
        y = df[target_col].values
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Linear Regression
        linreg = LinearRegression().fit(X_train, y_train)
        y_pred_lr = linreg.predict(X_test)

        # SVR
        svr_model = SVR(kernel='rbf', C=100, epsilon=0.1).fit(X_train, y_train)
        y_pred_svr = svr_model.predict(X_test)

        # ANN
        def build_ann_model(input_dim):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            return model

        ann_model = build_ann_model(X_train.shape[1])
        ann_model.fit(X_train, y_train, epochs=150, verbose=0)
        y_pred_ann = ann_model.predict(X_test).flatten()

        # Plot
        models = {
            'Linear Regression': y_pred_lr,
            'SVR': y_pred_svr,
            'ANN': y_pred_ann
        }

        pdf_buffer = BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(y_test, label="True", color="black", linewidth=2)
            for name, preds in models.items():
                ax.plot(preds, label=name)
            ax.set_title("📈 True vs Predicted Concentration")
            ax.legend()
            ax.set_ylabel("Concentration")
            ax.set_xlabel("Test Sample Index")
            st.pyplot(fig)
            pdf.savefig(fig)

        # Metrics
        st.markdown("### 📊 Model Performance")
        for name, preds in models.items():
            rmse = mean_squared_error(y_test, preds, squared=False)
            r2 = r2_score(y_test, preds)
            st.write(f"**{name}** — RMSE: `{rmse:.4f}`, R²: `{r2:.4f}`")

        pdf_buffer.seek(0)
        st.download_button("📥 Download Plot PDF", pdf_buffer, file_name="concentration_predictions.pdf")
