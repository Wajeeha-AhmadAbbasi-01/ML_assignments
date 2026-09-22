from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib


# =========================================================
# 1. LOAD TRAINED PIPELINE
# =========================================================

model = joblib.load(
    "churn_pipeline.joblib"  
)


# =========================================================
# 2. CREATE FASTAPI APP
# =========================================================

app = FastAPI(
    title="Telco Customer Churn API",
    description="API for predicting customer churn probability",
    version="1.0"
)


# =========================================================
# 3. INPUT DATA MODEL
# =========================================================

class CustomerRecord(BaseModel):

    tenure: float = Field(
        ge=0,
        description="Customer tenure in months"
    )

    MonthlyCharges: float = Field(
        ge=0,
        description="Customer monthly charges"
    )

    TotalCharges: float = Field(
        ge=0,
        description="Customer total charges"
    )

    Contract: str = Field(
        min_length=1,
        description="Customer contract type"
    )


# =========================================================
# 4. HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================================================
# 5. PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(record: CustomerRecord):

    # Convert API input into DataFrame
    customer = pd.DataFrame([
        {
            "Tenure Months": record.tenure,
            "Monthly Charges": record.MonthlyCharges,
            "Total Charges": record.TotalCharges,
            "Contract": record.Contract
        }
    ])

    # Predict churn probability
    probability = model.predict_proba(
        customer
    )[0, 1]

    return {
        "churn_probability": float(probability)
    }