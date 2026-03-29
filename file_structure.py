import os

folders = [
    "credit-risk-scoring/data/raw",
    "credit-risk-scoring/data/processed",
    "credit-risk-scoring/notebooks",
    "credit-risk-scoring/src",
    "credit-risk-scoring/models",
    "credit-risk-scoring/reports/figures",
]

files = [
    "credit-risk-scoring/notebooks/01_eda.ipynb",
    "credit-risk-scoring/src/data_preprocessing.py",
    "credit-risk-scoring/src/feature_engineering.py",
    "credit-risk-scoring/src/train.py",
    "credit-risk-scoring/src/evaluate.py",
    "credit-risk-scoring/requirements.txt",
    "credit-risk-scoring/README.md",
    "credit-risk-scoring/.gitignore",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for file in files:
    open(file, "w").close()