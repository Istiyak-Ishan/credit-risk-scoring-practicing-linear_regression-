from sklearn.metrics import roc_auc_score
def evaluate(model, X_test, y_test):
    results = {}
    for name, model in model.items():
        probs = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, probs)
        gini = 2* auc-1
        results[name]={
            'AUC':round(auc,4),
            'Gini':round(gini,4)
        }
    return results