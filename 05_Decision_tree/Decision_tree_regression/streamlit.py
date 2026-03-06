import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Decision Tree Regression Demo", layout="centered")

st.title("🌳 Decision Tree Regression – Interactive Visualization")
st.write("Adjust **Tree Depth and Noise** to see how the model fits the data.")

# ======================
# Sidebar Controls
# ======================

st.sidebar.header("Model Controls")

max_depth = st.sidebar.slider(
    "Max Depth of Tree",
    min_value=1,
    max_value=10,
    value=3
)

noise = st.sidebar.slider(
    "Noise Level",
    min_value=0.0,
    max_value=0.5,
    value=0.1
)

test_size = st.sidebar.slider(
    "Test Size (%)",
    min_value=10,
    max_value=50,
    value=20
)

# ======================
# Generate Dataset
# ======================

np.random.seed(42)

X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, noise, X.shape[0])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)

# ======================
# Model Training
# ======================

model = DecisionTreeRegressor(max_depth=max_depth)
model.fit(X_train, y_train)

# Prediction for smooth curve
X_plot = np.linspace(0, 5, 500).reshape(-1, 1)
y_plot = model.predict(X_plot)

# Test prediction
y_pred = model.predict(X_test)

# ======================
# Metrics
# ======================

mse = mean_squared_error(y_test, y_pred)

# ======================
# Plot
# ======================

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(X_train, y_train, label="Train Data")
ax.scatter(X_test, y_test, marker="x", label="Test Data")

ax.plot(X_plot, y_plot, linewidth=2, label=f"Decision Tree (depth={max_depth})")

ax.set_title("Decision Tree Regression Fit")
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend()

st.pyplot(fig)

# ======================
# Performance
# ======================

st.subheader("📊 Model Performance")

st.metric("Test MSE", f"{mse:.4f}")

# ======================
# Explanation
# ======================

st.markdown(
"""
### 🧠 Interpretation

- **Low depth (1–2)** → Underfitting (model too simple)  
- **Medium depth (3–5)** → Balanced model  
- **High depth (6+)** → Overfitting (model memorizes data)

Decision Trees split the feature space into **regions** and predict the **average value** in each region.
"""
)