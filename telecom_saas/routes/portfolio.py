"""
Portfolio route — POST /batch_predict
Enhanced with risk distribution buckets.
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File

import pandas as pd
import joblib
from pathlib import Path

from models.schemas import BatchPredictionResponse
from services.predict import predict_churn
from services.business_logic import calculate_ltv_revenue_at_risk

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Portfolio"])

BASE_DIR = Path(__file__).resolve().parent.parent
churn_rate = joblib.load(BASE_DIR / "artifacts" / "churn_rate.pkl")


@router.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)

        # Fill NaN
        string_cols = df.select_dtypes(include="object").columns
        numeric_cols = df.select_dtypes(include="number").columns
        df[string_cols] = df[string_cols].fillna("No")
        df[numeric_cols] = df[numeric_cols].fillna(0)

        results = []

        for _, row in df.iterrows():
            customer_dict = row.to_dict()
            prediction_result = predict_churn(customer_dict)

            probability = prediction_result["churn_probability"]
            risk_level = prediction_result["risk_level"]

            ltv_revenue_at_risk = calculate_ltv_revenue_at_risk(
                probability=probability,
                monthly_charge=customer_dict["Monthly Charge"],
                tenure=customer_dict["Tenure in Months"],
                churn_rate=churn_rate,
            )

            results.append({
                "churn_probability": probability,
                "risk_level": risk_level,
                "ltv_revenue_at_risk": ltv_revenue_at_risk,
            })

        results_df = pd.DataFrame(results)

        total_revenue_at_risk = results_df["ltv_revenue_at_risk"].sum()
        high_pct = (results_df["risk_level"] == "High").mean() * 100
        med_pct = (results_df["risk_level"] == "Medium").mean() * 100
        low_pct = (results_df["risk_level"] == "Low").mean() * 100

        top_10 = (
            results_df
            .sort_values("ltv_revenue_at_risk", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )

        # --- Risk Distribution Buckets ---
        low_mask = results_df["churn_probability"] < 0.3
        med_mask = (results_df["churn_probability"] >= 0.3) & (results_df["churn_probability"] < 0.7)
        high_mask = results_df["churn_probability"] >= 0.7

        total = len(results_df)
        risk_distribution = [
            {
                "bucket": "Low (0–0.3)",
                "count": int(low_mask.sum()),
                "percentage": round(float(low_mask.mean() * 100), 1),
                "revenue_at_risk": round(float(results_df.loc[low_mask, "ltv_revenue_at_risk"].sum()), 2),
            },
            {
                "bucket": "Medium (0.3–0.7)",
                "count": int(med_mask.sum()),
                "percentage": round(float(med_mask.mean() * 100), 1),
                "revenue_at_risk": round(float(results_df.loc[med_mask, "ltv_revenue_at_risk"].sum()), 2),
            },
            {
                "bucket": "High (0.7–1.0)",
                "count": int(high_mask.sum()),
                "percentage": round(float(high_mask.mean() * 100), 1),
                "revenue_at_risk": round(float(results_df.loc[high_mask, "ltv_revenue_at_risk"].sum()), 2),
            },
        ]

        return {
            "total_customers": len(results_df),
            "total_revenue_at_risk": round(total_revenue_at_risk, 2),
            "high_risk_percentage": round(high_pct, 2),
            "medium_risk_percentage": round(med_pct, 2),
            "low_risk_percentage": round(low_pct, 2),
            "risk_distribution": risk_distribution,
            "top_10_high_value_risky_customers": top_10,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
