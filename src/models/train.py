import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.evaluation.evaluate import evaluate
from src.data.ingest import load_raw_data
from src.features.build_features import build_feature_pipeline

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
import mlflow.sklearn
from sklearn.metrics import roc_auc_score


# =========================
# PATH CONFIG
# =========================
DATA_DIR = Path("data")
FEATURE_PATH = DATA_DIR / "processed" / "features.csv"

MODEL_DIR = Path("models")
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# =========================
# Split Data
# =========================

def split_data(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# =========================
# MODEL
# =========================
def build_models():
    return {
        "baseline": LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced"
        ),
        "rf": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "xgb": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric="logloss"
        )
    }


def train_models(models, X_train, y_train, X_test, y_test):

    trained_models = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            # train
            model.fit(X_train_scaled, y_train)

            # evaluate simple metric (for logging)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_proba)

            # logging
            mlflow.log_param("model_type", name)
            mlflow.log_metric("auc", auc)
            mlflow.sklearn.log_model(model, name)

            trained_models[name] = model

            print(f"Trained & Logged: {name}")

    return trained_models, scaler


# =========================
# SAVE
# =========================
def save_artifacts(models, scaler):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # simpan semua model dulu (selection nanti di evaluate/register)
    for name, model in models.items():
        path = MODEL_DIR / f"{name}.pkl"
        joblib.dump(model, path)

    joblib.dump(scaler, SCALER_PATH)


# =========================
# MAIN
# =========================
def main():

   # ===== ingestion =====
    df_raw = load_raw_data()

    # ===== feature =====
    df = build_feature_pipeline()

    print(df.shape)
    print(df["Churn"].value_counts(normalize=True))

    # ===== split =====
    X_train, X_test, y_train, y_test = split_data(df)

    # ===== train =====
    models = build_models()
    trained_models, scaler = train_models(
    models,
    X_train,
    y_train,
    X_test,
    y_test
)

    # ===== evaluate =====
    results = evaluate(trained_models, scaler, X_test, y_test)

    # ===== save =====
    save_artifacts(trained_models, scaler)

    print("\nFinal Decision:")
    print(results)


if __name__ == "__main__":
    main()