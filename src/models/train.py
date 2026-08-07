import joblib, time
from pathlib import Path
import mlflow.pyfunc
import numpy as np
import os
import mlflow

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
from src.features.build_features import FeatureBuilder
from src.evaluation.xai import log_shap_to_mlflow
from src.evaluation.fairness import evaluate_and_log_fairness, mitigate_bias

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import mlflow.sklearn

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(TRACKING_URI)


class SklearnProbaWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

# =========================
# CONFIG
# =========================
MODEL_DIR = Path("models")

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
# CROSS VALIDATION (FULL PIPELINE)
# =========================
def cross_validate_models(models, X, y, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = {name: [] for name in models.keys()}

    for train_idx, val_idx in skf.split(X, y):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        for name, model in models.items():

            pipeline = Pipeline([
                ("features", FeatureBuilder()),
                ("scaler", StandardScaler()),
                ("model", clone(model))
            ])

            pipeline.fit(X_train, y_train)
            y_proba = pipeline.predict_proba(X_val)[:, 1]

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
# TRAIN FULL (FULL PIPELINE)
# =========================
def train_models(models, X, y):
    trained_models = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("features", FeatureBuilder()),
            ("scaler", StandardScaler()),
            ("model", model)
        ])

        pipeline.fit(X, y)
        trained_models[name] = pipeline

    return trained_models

# =========================
# SAVE
# =========================
def save_best_artifacts(best_model, threshold):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_DIR / "best_pipeline.pkl")
    joblib.dump(threshold, MODEL_DIR / "threshold.pkl")

# =========================
# MAIN
# =========================
def main():

    # ===== data =====
    df = load_raw_data()

    X = df.drop("Churn", axis=1)
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # ===== holdout =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ===== save reference data for drift =====
    X_train.to_csv("models/reference_data.csv", index=False)

    # ===== CV =====
    models = build_models()
    cv_results = cross_validate_models(models, X, y)

    best_model_name = max(
        cv_results,
        key=lambda x: cv_results[x]["business_score"]
    )

    # ===== train full =====
    models = build_models()
    trained_models = train_models(models, X, y)

    # ===== evaluate =====
    results = evaluate(trained_models, X_test, y_test)

    best_model = results[best_model_name]["model"]
    best_threshold = cv_results[best_model_name]["threshold"]

    # ===== sanity check =====
    preds = best_model.predict(X_test)
    print("Sample preds:", preds[:5])

    # ===== MLflow =====
    run_ids = {}
    
    # Define sensitive feature name based on your raw dataset schema
    SENSITIVE_FEATURE_NAME = "SeniorCitizen"
    
    # --- NEW ADDITION: EXECUTE BIAS MITIGATION HERE ---
    print(f"\n[INFO] Mitigating bias on Champion Model ({best_model_name})...")
    
    mitigated_model = mitigate_bias(
        champion_model=best_model, 
        X_train=X_train, 
        y_train=y_train, 
        sensitive_features=X_train[SENSITIVE_FEATURE_NAME]
    )
    # --------------------------------------------------

    for name, metrics in results.items():

        pipeline = trained_models[name]
        cv_metrics = cv_results[name]
        
        # Isolate the exact optimized threshold from the evaluation results
        model_threshold = metrics.get("threshold", 0.5)

        signature = infer_signature(
            X_train,
            pipeline.predict(X_train)
        )

        with mlflow.start_run(run_name=name):

            mlflow.log_metric("auc", metrics["auc"])
            mlflow.log_metric("f1", metrics["f1"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])
            mlflow.log_metric("business_score", metrics["business_score"])

            mlflow.log_metric("cv_auc", cv_metrics["auc"])
            mlflow.log_metric("cv_auc_std", cv_metrics["auc_std"])
            mlflow.log_metric("cv_f1", cv_metrics["f1"])
            mlflow.log_metric("cv_f1_std", cv_metrics["f1_std"])
            mlflow.log_metric("cv_business", cv_metrics["business_score"])
            mlflow.log_metric("cv_business_std", cv_metrics["business_std"])
            joblib.dump(pipeline, "models/temp_pipeline.pkl")
            
            mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SklearnProbaWrapper(),
            artifacts={
                "model": "models/temp_pipeline.pkl"
            },
            input_example=X_train.iloc[:5]
            )

            log_shap_to_mlflow(pipeline=pipeline, X_train=X_train, run_name=name)
            
            # ==========================================
            # FAIRNESS EVALUATION INTEGRATION
            # ==========================================
            if SENSITIVE_FEATURE_NAME in X_test.columns:
                
                # 1. Extract the sensitive column directly from holdout test data
                sensitive_feature_test = X_test[SENSITIVE_FEATURE_NAME]
                
                # 2. Generate predictions and enforce thresholds
                if name == best_model_name:
                    # CHAMPION: Use only champion tags model
                    # (ThresholdOptimizer only output 0 or 1)
                    y_pred_binary = mitigated_model.predict(
                        X_test, 
                        sensitive_features=sensitive_feature_test
                    )
                else:
                    # LOSERS: use old Treshold implementation)
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                    y_pred_binary = (y_proba >= model_threshold).astype(int)
                    
                # 3. Execute fairness evaluation and log synchronously to MLflow
                evaluate_and_log_fairness(
                    y_true=y_test,
                    y_pred=y_pred_binary,
                    sensitive_feature=sensitive_feature_test,
                    feature_name=SENSITIVE_FEATURE_NAME
                )
            else:
                print(f"WARNING: Sensitive feature '{SENSITIVE_FEATURE_NAME}' not found in X_test. Skipping fairness audit.")

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

    # ===== save =====
    save_best_artifacts(mitigated_model, best_threshold)

if __name__ == "__main__":
    main()


