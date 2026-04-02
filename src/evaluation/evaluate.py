from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np


def evaluate(models, scaler, X_test, y_test):
    """
    Evaluate multiple models and select best one
    """

    X_test_scaled = scaler.transform(X_test)

    results = {}

    for name, model in models.items():

        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        best_t, best_f1 = find_best_threshold(y_test, y_proba)
        y_pred = (y_proba >= best_t).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "auc": auc,
            "f1": best_f1,
            "precision": precision,
            "recall": recall,
            "threshold": best_t
        }

        print(f"\n=== {name.upper()} ===")
        print(f"AUC       : {auc:.4f}")
        print(f"F1        : {best_f1:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"Threshold : {best_t:.2f}")

    return results


def find_best_threshold(y_true, y_proba):
    best_t = 0.5
    best_f1 = 0

    for t in np.linspace(0.1, 0.9, 17):
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1