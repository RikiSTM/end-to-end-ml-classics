# app/main.py
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from app.adapter import build_full_input
from app.logger import setup_logger
from app.model_loader import load_artifacts
from app.schema import ChurnRequest, ChurnResponse

logger = setup_logger()
app = FastAPI()
model = load_artifacts()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict", response_model=ChurnResponse)
def predict(data: ChurnRequest):
    try:
        start_time = datetime.now(timezone.utc)
        
        input_data = data.model_dump()
        full_data = build_full_input(input_data)
        df = pd.DataFrame([full_data])
        
        # ===== log inference for Data drift =====
        log_path = Path("logs/inference_log.csv")
        # checking folder log
        log_path.parent.mkdir(parents=True, exist_ok=True) 
        df.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
        
        # ===== model predict (Fairlearn ThresholdOptimizer) =====
        sensitive_feature = df[["SeniorCitizen"]]
        
        # add sensitive features to pred
        pred = int(model.predict(df, sensitive_features=sensitive_feature)[0])
        proba = float(pred) 
        
        latency = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(f"pred | proba={proba:.3f} pred={pred} latency={latency:.4f}s")
        
        return {
            "churn_probability": proba,
            "churn_prediction": pred
        }
    except Exception as e:
        logger.error(f"internal_error | {e!s}")
        raise 