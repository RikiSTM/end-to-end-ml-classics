from pathlib import Path

import joblib

MODEL_PATH = Path("models/best_pipeline.pkl")

model = None

def load_artifacts():
    global model
    if model is None:
        # Load langsung dari Joblib, bukan MLflow
        model = joblib.load(MODEL_PATH)
    return model