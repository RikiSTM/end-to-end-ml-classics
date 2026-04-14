from fastapi import FastAPI
from mlflow import data
from app.model_loader import load_artifacts
from app.schema import ChurnRequest, ChurnResponse
import pandas as pd
from datetime import datetime, timezone
from app.logging import setup_logger
from fastapi import Request
from fastapi.responses import JSONResponse
from app.adapter import build_full_input
from pathlib import Path


logger = setup_logger()
start_time = datetime.now(timezone.utc)



app = FastAPI()

model, threshold = load_artifacts()

@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(data: ChurnRequest):
    try:
        input_data = data.model_dump()
        full_data = build_full_input(input_data)

        df = pd.DataFrame([full_data])

        # ===== log inference for Data drift =====
        log_path = Path("logs/inference_log.csv")

        df.to_csv(
            log_path,
            mode="a",
            header=not log_path.exists(),
            index=False
        )
                

        # ===== model predict =====
        proba = model.predict(df)[0]
        pred = int(proba >= threshold)

        latency = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.info(
            f"pred | proba={proba:.3f} pred={pred} threshold={threshold} latency={latency:.4f}s"
        )

        return {
            "churn_probability": proba,
            "threshold": threshold,
            "churn_prediction": pred
        }

    except Exception as e:
        raise e
    
app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):

    logger.error(f"internal_error | {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )