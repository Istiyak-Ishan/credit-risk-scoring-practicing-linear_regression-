from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_model(transformer, X_train, y_train):
    os.makedirs('models',exist_ok=True)
    models = {
        'logistic': Pipeline([
            ('transform',transformer),
            ('scaler',StandardScaler(with_mean=False)),
            ('model',LogisticRegression(max_iter=1000))
        ]),
        'svm': Pipeline([
            ('transform',transformer),
            ('scaler',StandardScaler(with_mean=False)),
            ('model',SVC(probability=True))
        ]),
        'knn': Pipeline([
            ('transform',transformer),
            ('scaler',StandardScaler(with_mean=False)),
            ('model',KNeighborsClassifier(n_neighbors=5))
        ])
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        joblib.dump(model,f'models/{name}.pkl')
    return models
