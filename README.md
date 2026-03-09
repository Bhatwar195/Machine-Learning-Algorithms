# Machine Learning Algorithms Implementation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)


A structured collection of fundamental machine learning algorithm implementations using Python and Scikit-Learn. Each module is presented through Jupyter Notebooks with accompanying visualizations and interactive Streamlit applications.

## Overview

This repository serves as a practical reference for core machine learning concepts, combining clean implementations with visual tools to build intuition around how each algorithm behaves.

---

## Repository Structure

### 01 — Linear Regression

Implementation of Linear Regression variants with a focus on regression evaluation metrics.

**Algorithms Covered**
- Simple Linear Regression
- Multiple Linear Regression
- Ridge Regression
- Lasso Regression

**Evaluation Metrics**
- MAE, MSE, RMSE, R²

**Datasets**
- Bangalore House Price Dataset
- Placement Dataset

---

### 02 — Logistic Regression

Implementation of Logistic Regression for binary classification tasks.

**Concepts Covered**
- Binary classification
- Sigmoid function and probability prediction
- Model evaluation and threshold tuning

---

### 03 — K-Nearest Neighbors (KNN)

Implementation of KNN for both classification and regression use cases, with an interactive Streamlit visualization.

**Implementations**
- Breast cancer classification
- House price prediction (KNN Regression)
- Streamlit visualization app

**Key Concepts**
- Distance-based learning
- Optimal K selection
- Feature scaling

---

### 04 — Support Vector Machine (SVM)

Implementation of SVM for classification and regression, including decision boundary visualization.

**Implementations**
- SVM classification
- Support Vector Regression (SVR)
- Streamlit decision boundary visualization

**Key Concepts**
- Maximum margin hyperplane
- Kernel trick
- Support vectors
- Regularization parameter (C)

---

### 05 — Decision Tree

In-depth implementation and visualization of Decision Tree algorithms for both classification and regression.

**Implementations**
- Decision Tree classification
- Decision Tree regression
- Overfitting vs. underfitting demonstration
- Tree visualization using `dtreeviz`
- Streamlit apps for classification and regression

**Key Concepts**
- Gini impurity and information gain
- Tree depth control
- Model interpretability

---

## Technologies

| Category | Tools |
|---|---|
| Language | Python |
| ML Framework | Scikit-Learn |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib, dtreeviz |
| Interactive Apps | Streamlit |

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch a Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Run a Streamlit app (example):
   ```bash
   streamlit run 03_KNN/app.py
   ```

---

## Purpose

This repository is intended for learning and demonstration purposes. The goal is to practice implementing core machine learning algorithms from the ground up, understand their mechanics through visualization, and develop intuition for when and how to apply each method.
