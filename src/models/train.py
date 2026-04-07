import joblib, time
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.evaluation.evaluate import evaluate, find_best_threshold_business
from src.data.ingest import load_raw_data
from src.features.build_features import build_feature_pipeline

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")


# =========================
# CONFIG
# =========================
MODEL_DIR = Path("models")
SCALER_PATH = MODEL_DIR / "scaler.pkl"


# =========================
# SPLIT
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
# MODELS
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


# =========================
# TRAIN FULL DATA
# =========================
def train_models(models, X, y):
    trained_models = {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for name, model in models.items():
        model.fit(X_scaled, y)
        trained_models[name] = model

    return trained_models, scaler


# =========================
# CROSS VALIDATION
# =========================
def cross_validate_models(models, X, y, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_results = {name: [] for name in models.keys()}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for name, model in models.items():

            m = clone(model)
            m.fit(X_train_scaled, y_train)

            y_proba = m.predict_proba(X_val_scaled)[:, 1]

            best_t, best_business = find_best_threshold_business(y_val, y_proba)
            y_pred = (y_proba >= best_t).astype(int)

            auc = roc_auc_score(y_val, y_proba)
            f1 = f1_score(y_val, y_pred)

            cv_results[name].append({
                "auc": auc,
                "f1": f1,
                "business_score": best_business,
                "threshold": best_t
            })

    # aggregate
    final_results = {}

    for name, folds in cv_results.items():

        final_results[name] = {
            "auc": np.mean([f["auc"] for f in folds]),
            "auc_std": np.std([f["auc"] for f in folds]),

            "f1": np.mean([f["f1"] for f in folds]),
            "f1_std": np.std([f["f1"] for f in folds]),

            "business_score": np.mean([f["business_score"] for f in folds]),
            "business_std": np.std([f["business_score"] for f in folds]),

            "threshold": np.mean([f["threshold"] for f in folds])
        }

    return final_results


# =========================
# SAVE
# =========================
def save_artifacts(models, scaler):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        joblib.dump(model, MODEL_DIR / f"{name}.pkl")

    joblib.dump(scaler, SCALER_PATH)


# =========================
# MAIN
# =========================
def main():

    # ===== data =====
    load_raw_data()
    df = build_feature_pipeline()

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # ===== holdout =====
    X_train, X_test, y_train, y_test = split_data(df)

    # ===== CV =====
    models = build_models()
    cv_results = cross_validate_models(models, X, y)

    best_model_name = max(
        cv_results,
        key=lambda x: cv_results[x]["business_score"]
    )

    # ===== train full =====
    models = build_models()
    trained_models, scaler = train_models(models, X, y)

    # ===== evaluate holdout =====
    results = evaluate(trained_models, scaler, X_test, y_test)

    # ===== MLflow =====
    run_ids = {}

    for name, metrics in results.items():

        model = trained_models[name]
        cv_metrics = cv_results[name]

        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", model)
        ])

        signature = infer_signature(
            X_train,
            pipeline.predict(X_train)
        )

        with mlflow.start_run(run_name=name):

            mlflow.set_tag("model_family", "churn")
            mlflow.set_tag("features", ",".join(X_train.columns))
            mlflow.set_tag("notes", "baseline churn model with scaling + threshold tuning")

            # holdout
            mlflow.log_metric("auc", metrics["auc"])
            mlflow.log_metric("f1", metrics["f1"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("business_score", metrics["business_score"])

            # CV
            mlflow.log_metric("cv_auc", cv_metrics["auc"])
            mlflow.log_metric("cv_auc_std", cv_metrics["auc_std"])

            mlflow.log_metric("cv_f1", cv_metrics["f1"])
            mlflow.log_metric("cv_f1_std", cv_metrics["f1_std"])

            mlflow.log_metric("cv_business", cv_metrics["business_score"])
            mlflow.log_metric("cv_business_std", cv_metrics["business_std"])

            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:5]
            )

            run_ids[name] = mlflow.active_run().info.run_id

    # ===== registry =====
    client = MlflowClient()
    model_name = "churn_model"

    best_run_id = run_ids[best_model_name]
    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    # wait ready
    for _ in range(10):
        mv = client.get_model_version(name=model_name, version=result.version)
        if mv.status == "READY":
            break
        time.sleep(1)

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=result.version
    )

    # save local
    save_artifacts(trained_models, scaler)

    print("\nFINAL CV RESULTS:")
    print(cv_results)


if __name__ == "__main__":
    main()