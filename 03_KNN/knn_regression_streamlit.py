import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="KNN Regression Lab", layout="centered")

st.title("🔵 KNN Regression – Interactive ML Lab")
st.write("Experiment with **K, Noise, and Weighting Strategy** to understand Bias-Variance tradeoff.")

# ==============================
# Sidebar Controls
# ==============================

st.sidebar.header("Model Controls")

k = st.sidebar.slider("Number of Neighbors (K)", 1, 25, 5)

noise = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1)

weights = st.sidebar.selectbox(
    "Weight Function",
    ["uniform", "distance"]
)

test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)

# ==============================
# Generate Non-Linear Dataset
# ==============================

np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, noise, X.shape[0])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)

# ==============================
# Model Training
# ==============================

model = KNeighborsRegressor(n_neighbors=k, weights=weights)
model.fit(X_train, y_train)

# Prediction
X_plot = np.linspace(0, 5, 400).reshape(-1, 1)
y_plot = model.predict(X_plot)
y_pred_test = model.predict(X_test)

# ==============================
# Metrics
# ==============================

mse = mean_squared_error(y_test, y_pred_test)

# ==============================
# Plotting
# ==============================

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(X_train, y_train, label="Train Data", alpha=0.7)
ax.scatter(X_test, y_test, label="Test Data", marker="x")
ax.plot(X_plot, y_plot, linewidth=2, label=f"KNN Prediction (K={k})")

ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title("KNN Regression – Bias vs Variance")
ax.legend()

st.pyplot(fig)

# ==============================
# Results Section
# ==============================

st.subheader("📊 Model Performance")
st.metric(label="Test MSE", value=f"{mse:.4f}")

st.markdown(
    """
### 🧠 Interpretation Guide

- **Small K (1-3)** → Low bias, High variance → Overfitting
- **Medium K (5-10)** → Balanced model
- **Large K (>15)** → High bias, Low variance → Underfitting
- **Distance weights** → Closer neighbors influence more
- **Uniform weights** → All neighbors treated equally
"""
)

st.info("Try increasing noise and decreasing K to clearly observe overfitting.")