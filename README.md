# Credit Risk Scoring — sklearn ML Pipeline

A modular credit risk scoring pipeline built with Python. Practices and compares classification approaches — Logistic Regression, SVM, KNN — with regularization and probability calibration using a clean scikit-learn pipeline structure.

## Models Included
- **Logistic Regression** (with L1/L2 regularization)
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Probability Calibration** (Platt scaling / isotonic regression)

## Project Structure
```
credit-risk-scoring/
├── data/
│   ├── raw/                    # Original input data
│   └── processed/              # Cleaned & feature-engineered data
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory data analysis
├── src/
│   ├── data_preprocessing.py   # Data cleaning & preparation
│   ├── feature_engineering.py  # Feature creation & encoding
│   ├── train.py                # Model training
│   └── evaluate.py             # Metrics & evaluation
├── models/                     # Saved model artifacts
├── reports/
│   └── figures/                # Plots & visualizations
├── requirements.txt
└── .gitignore
```

## Quickstart
```bash
git clone https://github.com/Istiyak-Ishan/credit-risk-scoring-practicing-linear_regression-.git
cd credit-risk-scoring-practicing-linear_regression-

pip install -r requirements.txt

# Explore data
jupyter notebook credit-risk-scoring/notebooks/01_eda.ipynb

# Train model
python credit-risk-scoring/src/train.py

# Evaluate
python credit-risk-scoring/src/evaluate.py
```

## Stack
`scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `jupyter`
