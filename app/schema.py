from pydantic import BaseModel


class ChurnRequest(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    SeniorCitizen: int


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: int