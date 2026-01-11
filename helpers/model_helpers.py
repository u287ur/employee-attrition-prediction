from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
import numpy

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "classification_report": classification_report(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "recall_1": recall_score(y_test, y_pred),
        "f1_1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    
def threshold_search(y_true, y_proba, thresholds):
    results = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)

        results.append({
            "threshold": f"{t:0.2f}",
            "precision": f"{precision_score(y_true, y_pred_t):0.5f}",
            "recall": f"{recall_score(y_true, y_pred_t):0.5f}",
            "f1": f"{f1_score(y_true, y_pred_t):0.5f}"
        })

    return results

def threshold_cv_recall(model, X, y, threshold=0.4, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    recalls = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        recalls.append(recall_score(y_test, y_pred))

    return numpy.mean(recalls), numpy.std(recalls)