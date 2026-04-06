import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


MODEL_PATH = "models/churn_pipeline.pkl"

app = FastAPI(title="Telecom Churn Prediction API")

pipeline = joblib.load(MODEL_PATH)


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Telecom Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    input_df = pd.DataFrame([customer.model_dump()])
    prediction = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    return {
        "churn_prediction": prediction,
        "churn_label": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
    }