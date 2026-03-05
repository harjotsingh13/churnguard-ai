"""
Strategy Simulation route — POST /simulate_strategy
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File

import pandas as pd
import joblib
from pathlib import Path

from models.schemas import SimulationResponse
from services.predict import predict_churn
from services.business_logic import calculate_ltv_revenue_at_risk

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Strategy"])

BASE_DIR = Path(__file__).resolve().parent.parent
churn_rate = joblib.load(BASE_DIR / "artifacts" / "churn_rate.pkl")


@router.post("/simulate_strategy", response_model=SimulationResponse)
async def simulate_strategy(
    file: UploadFile = File(...),
    discount: float = 0,
):
    try:
        df = pd.read_csv(file.file)

        # Fill NaN
        string_cols = df.select_dtypes(include="object").columns
        numeric_cols = df.select_dtypes(include="number").columns
        df[string_cols] = df[string_cols].fillna("No")
        df[numeric_cols] = df[numeric_cols].fillna(0)

        original_total = 0
        new_total = 0

        for _, row in df.iterrows():
            customer_dict = row.to_dict()
            prediction_result = predict_churn(customer_dict)
            probability = prediction_result["churn_probability"]

            original_risk = calculate_ltv_revenue_at_risk(
                probability=probability,
                monthly_charge=customer_dict["Monthly Charge"],
                tenure=customer_dict["Tenure in Months"],
                churn_rate=churn_rate,
            )
            original_total += original_risk

            adjusted_charge = max(customer_dict["Monthly Charge"] - discount, 0)

            new_risk = calculate_ltv_revenue_at_risk(
                probability=probability,
                monthly_charge=adjusted_charge,
                tenure=customer_dict["Tenure in Months"],
                churn_rate=churn_rate,
            )
            new_total += new_risk

        return {
            "original_revenue_at_risk": round(original_total, 2),
            "revenue_after_strategy": round(new_total, 2),
            "revenue_saved": round(original_total - new_total, 2),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
