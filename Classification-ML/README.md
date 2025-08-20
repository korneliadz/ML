ML

This repository contains two machine learning projects based on Kaggle datasets, covering both classification and regression tasks.

Repository Structure

Each project follows the same structure:

attachments.py – helper functions for data input/output and utility handling.

classes_and_functions.py – implementation of custom classes and reusable ML functions.

workspace.py – main workspace script for running the ML pipeline.

classification/ – Apple Quality Dataset

Goal: Predict apple quality (good vs bad) using classification models.

Dataset on Kaggle

regression/ – House Prices Dataset

Goal: Predict house sale prices using regression models.

Dataset on Kaggle

Features

End-to-end ML workflows: data preprocessing, feature engineering, model training, and evaluation.

Uses popular ML libraries: pandas, NumPy, scikit-learn, matplotlib, and seaborn.

Compares different algorithms with performance metrics (e.g., accuracy, precision/recall for classification; RMSE, MAE for regression).

Includes visualizations for exploratory data analysis and model evaluation.

Requirements

Python 3.8+

NumPy

pandas

scikit-learn

matplotlib

seaborn

Install dependencies with:

pip install -r requirements.txt

Usage

For classification (Apple Quality):

cd classification
python workspace.py


For regression (House Prices):

cd regression
python workspace.py


⚠️ Before running, download the datasets from Kaggle and place the CSV files in the appropriate project folder.

License
