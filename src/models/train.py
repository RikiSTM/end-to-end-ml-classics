import joblib, time
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
from mlflow.models import infer_signature
mlflow.set_tracking_uri("http://127.0.0.1:5000")
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline



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


def train_models(models, X_train, y_train):

    trained_models = {}

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    for name, model in models.items():

        model.fit(X_train_scaled, y_train)

        trained_models[name] = model

        print(f"Trained: {name}")

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
    load_raw_data()

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
    y_train
)

    # ===== evaluate =====
    results = evaluate(trained_models, scaler, X_test, y_test)
    best_model_name = max(results, key=lambda x: results[x]["f1"])
    best_result = results[best_model_name]

    # ===== ML Flow Log =====

    run_ids = {}

    for name, metrics in results.items():

        model = trained_models[name]

        # gabungkan scaler + model
        pipeline = Pipeline([
            ("scaler", scaler),
            ("model", model)
        ])

        # input contoh (RAW)
        input_example = X_train.iloc[:5]

        # signature harus pakai pipeline, bukan model
        signature = infer_signature(
            X_train,
            pipeline.predict(X_train)
        )

        with mlflow.start_run(run_name=name) as run:


            mlflow.set_tag("model_family", "churn")
            mlflow.set_tag("model_name", name)
            mlflow.set_tag("stage_candidate", "true")

            mlflow.set_tag("features", ",".join(X_train.columns))
            mlflow.set_tag("target", "Churn")

            mlflow.set_tag("notes", "baseline churn model with scaling + threshold tuning")
            mlflow.log_param("model_type", name)
            mlflow.log_param("threshold", metrics["threshold"])

            mlflow.log_metric("auc", metrics["auc"])
            mlflow.log_metric("f1", metrics["f1"])
            mlflow.log_metric("precision", metrics["precision"])
            mlflow.log_metric("recall", metrics["recall"])

            mlflow.sklearn.log_model(
                pipeline,   # penting: pipeline, bukan model
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

            run_ids[name] = run.info.run_id
                


    # ======== MLflow Registry =======

    client = MlflowClient()
    model_name = "churn_model"
    best_run_id = run_ids[best_model_name]
    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
    )
    
    # ===== WAIT UNTIL READY =====
    for _ in range(10):
        model_version = client.get_model_version(
            name=model_name,
            version=result.version
        )
        if model_version.status == "READY":
            break
        time.sleep(1)
    else:
        raise Exception("Model not ready after waiting")
    

    # ===== TRANSITION =====
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=result.version
    )


    # ===== save =====
    save_artifacts(trained_models, scaler)

    print("\nFinal Decision:")
    print(results)


if __name__ == "__main__":
    main()