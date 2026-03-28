from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
import numpy as np
import pandas as pd


def evaluate(model,scaler, X_test, y_test):

    X_test_scaled = scaler.transform(X_test)    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Confusion Matrix: ")
    print(cm)

    # Sanity check test
    sanity_check(y_test)

      # Shuffle test
    shuffle_test(model, scaler, X_test, y_test)

    # Treshold tuning check
    threshold_tuning(model, scaler, X_test, y_test)


def sanity_check(y_test):

    # baseline: semua prediksi = kelas mayoritas
    majority_class = y_test.mode()[0]
    y_dummy = [majority_class] * len(y_test)

    acc = accuracy_score(y_test, y_dummy)

    print("=== Sanity Check ===")
    print(f"Dummy Accuracy (majority class): {acc:.4f}")

def shuffle_test(model, scaler, X_test, y_test):

    X_test_scaled = scaler.transform(X_test)

    # shuffle label
    y_shuffled = np.random.permutation(y_test)

    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_shuffled, y_proba)

    print("=== Shuffle Test ===")
    print(f"ROC-AUC (shuffled labels): {auc:.4f}")



def threshold_tuning(model, scaler, X_test, y_test):

    X_test_scaled = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print("=== Threshold Tuning ===")

    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:

        y_pred = (y_proba >= t).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"Threshold: {t}")
        print(f"Precision: {precision:.3f} | Recall: {recall:.3f}")
        print("---")