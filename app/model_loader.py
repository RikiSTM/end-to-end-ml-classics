import mlflow.pyfunc
import joblib
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
MODEL_URI = "models:/churn_model@champion"
THRESHOLD_PATH = "models/threshold.pkl"

model = None
threshold = None


def load_artifacts():
    global model, threshold

    if model is None:
        model = mlflow.pyfunc.load_model(MODEL_URI)

    if threshold is None:
        threshold = joblib.load(THRESHOLD_PATH)

    return model, threshold