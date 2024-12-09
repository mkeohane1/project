# DS5010 Project
# Simple Linear Regression Toolkit

## Overview
This Python package provides a collection of tools to streamline the end-to-end workflow of building, training, and evaluating linear regression models. It handles everything from data cleaning and preprocessing, through exploratory data analysis and feature selection, to model fitting, residual analysis, and evaluation.

The toolkit is designed to help data scientists and practitioners easily set up a robust linear regression pipeline without having to rewrite boilerplate code each time. Whether you're just starting with linear regression or looking to standardize your workflow, this package offers a clean, modular solution.

## Features
- **Data Cleaning and Preprocessing:**  
  - Impute missing values in numeric and categorical columns  
  - Detect and remove outliers based on the Interquartile Range (IQR)  
  - Encode categorical variables using one-hot or label encoding

- **Exploratory Data Analysis (EDA):**  
  - Compute common descriptive statistics (mean, median, variance, standard deviation)  
  - Calculate correlation coefficients to identify linear relationships

- **Feature Selection:**  
  - Implement backward elimination or forward selection to reduce dimensionality and improve model interpretability

- **Model Selection & Training:**  
  - Fit Ordinary Least Squares (OLS), Lasso, and Ridge regression models to compare different techniques and regularization strengths

- **Residual Analysis:**  
  - Examine residuals to detect patterns or violations of regression assumptions  
  - Visualize predicted vs. actual values

- **Model Evaluation:**  
  - Evaluate models using R², Mean Absolute Error (MAE), Mean Squared Error (MSE)  
  - Conduct cross-validation and derive confidence intervals for robust performance metrics

# Example Usage

import pandas as pd

from cleaning import preprocess_data


# Load your dataset
df = pd.read_csv('your_data.csv')

# Preprocess your data
cleaned_df = preprocess_data(
    df,
    numeric_strategy='mean',
    categorical_strategy='most_frequent',
    outlier_method='iqr',
    encode_method='one-hot'
)

# After preprocessing, proceed with EDA, feature selection, and model training

