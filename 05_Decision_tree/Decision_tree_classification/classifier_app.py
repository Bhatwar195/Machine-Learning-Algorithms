import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Decision Tree Classification Demo", layout="centered")

st.title("🌳 Decision Tree Classification – Interactive Visualization")
st.write("Adjust **Tree Depth and Noise** to see how the decision boundary changes.")

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
    value=0.2
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

X, y = make_moons(n_samples=300, noise=noise, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)

# ======================
# Train Model
# ======================

model = DecisionTreeClassifier(max_depth=max_depth)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ======================
# Decision Boundary
# ======================

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ======================
# Plot
# ======================

fig, ax = plt.subplots(figsize=(8,5))

ax.contourf(xx, yy, Z, alpha=0.3)

ax.scatter(
    X_train[:,0], X_train[:,1],
    c=y_train,
    label="Train Data"
)

ax.scatter(
    X_test[:,0], X_test[:,1],
    c=y_test,
    marker="x",
    label="Test Data"
)

ax.set_title(f"Decision Tree Decision Boundary (depth={max_depth})")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.legend()

st.pyplot(fig)

# ======================
# Performance
# ======================

st.subheader("📊 Model Performance")

st.metric("Accuracy", f"{accuracy:.3f}")

# ======================
# Explanation
# ======================

st.markdown(
"""
### 🧠 Interpretation

- **Low depth (1–2)** → Underfitting (simple boundary)  
- **Medium depth (3–5)** → Balanced model  
- **High depth (6+)** → Overfitting (complex boundary)

Decision Trees split the feature space into **rectangular regions** based on feature thresholds.
"""
)