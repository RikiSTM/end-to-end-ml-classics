from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)
import numpy as np


def evaluate(models, X_test, y_test):
    """
    Evaluate multiple models and select best one
    """

    results = {}

    for name, model in models.items():

    
        y_proba = model.predict_proba(X_test)[:, 1]

        best_t, best_business = find_best_threshold_business(y_test, y_proba)
        y_pred = (y_proba >= best_t).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "auc": auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "threshold": best_t,
            "business_score": best_business
        }

        print(f"\n=== {name.upper()} ===")
        print(f"AUC       : {auc:.4f}")
        print(f"F1        : {f1:.4f}")
        print(f"Business  : {best_business:.2f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"Threshold : {best_t:.2f}")

    return results


def find_best_threshold_business(y_true, y_proba):
    best_t = 0.5
    best_score = -1e9

    for t in np.linspace(0.1, 0.9, 17):
        y_pred = (y_proba >= t).astype(int)
        score = compute_business_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


def compute_business_score(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    return tp*100 - fp*75 - fn*200