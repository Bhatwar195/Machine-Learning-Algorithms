import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SVM Visualizer", layout="centered")

st.title("SVM Classification – Interactive Visualization")

st.write("Adjust parameters to see how SVM decision boundary changes.")

# Sidebar controls
kernel = st.sidebar.selectbox("Select Kernel", ("linear", "rbf", "poly"))
C = st.sidebar.slider("Regularization (C)", 0.1, 10.0, 1.0)
gamma = st.sidebar.selectbox("Gamma", ("scale", "auto"))

# Generate dataset
X, y = datasets.make_moons(n_samples=200, noise=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train SVM
model = SVC(kernel=kernel, C=C, gamma=gamma)
model.fit(X, y)

# Create mesh grid
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
fig, ax = plt.subplots()

ax.contourf(xx, yy, Z, alpha=0.3)
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

# Highlight support vectors
ax.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100,
    facecolors='none',
    edgecolors='red'
)

ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("SVM Decision Boundary")

st.pyplot(fig)