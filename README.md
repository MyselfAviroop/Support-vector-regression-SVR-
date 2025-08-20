# SVM & SVR Learning Projects

## Overview
This repository contains hands-on experiments with **Support Vector Machines (SVM)** and **Support Vector Regression (SVR)** using Python. The focus was on:

- Preprocessing categorical and numerical data
- Training SVR models with multiple kernels
- Hyperparameter tuning using GridSearchCV
- Visualizing high-dimensional data using PCA
- Exploring polynomial feature interactions
- 3D data visualization with Plotly

## Datasets Used
- `tips` dataset from Seaborn (regression)
- Generated 2D classification datasets (synthetic) for SVM visualizations

## Key Concepts Explored
1. **Label Encoding & OneHotEncoding** – Transform categorical data into numeric format.
2. **Feature Scaling** – Standardization to normalize numeric features.
3. **SVR Training** – Linear, RBF, and polynomial kernels.
4. **GridSearchCV** – Finding optimal hyperparameters (`C`, `gamma`, `kernel`).
5. **PCA** – Reducing dimensionality for visualization.
6. **Decision Boundary Visualization** – 2D and 3D plots for classification datasets.
7. **Polynomial Features** – Interaction terms for non-linear boundaries.

## Results
- Successfully trained SVR models with tuned hyperparameters.
- Evaluated models using **R²** and **MAE**.
- Created 2D PCA plots to approximate SVR hyperplanes.
- Built 3D scatter plots to visualize polynomial interactions.

## How to Run
1. Clone the repository
2. Install dependencies:  
   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn plotly
