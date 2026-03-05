"""
Prediction route — POST /predict
"""

import logging
from fastapi import APIRouter, HTTPException

from models.schemas import CustomerData, PredictionResponse
from services.predict import predict_churn
from services.business_logic import calculate_ltv_revenue_at_risk, recommend_action
from services.segmentation import predict_segment

import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

BASE_DIR = Path(__file__).resolve().parent.parent
churn_rate = joblib.load(BASE_DIR / "artifacts" / "churn_rate.pkl")


def _format_customer(customer_dict: dict) -> dict:
    """Map underscore field names to space-separated names expected by the model."""
    return {
        "Gender": customer_dict["Gender"],
        "Age": customer_dict["Age"],
        "Married": customer_dict["Married"],
        "Number of Dependents": customer_dict["Number_of_Dependents"],
        "Number of Referrals": customer_dict["Number_of_Referrals"],
        "Tenure in Months": customer_dict["Tenure_in_Months"],
        "Offer": customer_dict["Offer"],
        "Phone Service": customer_dict["Phone_Service"],
        "Avg Monthly Long Distance Charges": customer_dict["Avg_Monthly_Long_Distance_Charges"],
        "Multiple Lines": customer_dict["Multiple_Lines"],
        "Internet Service": customer_dict["Internet_Service"],
        "Internet Type": customer_dict["Internet_Type"],
        "Avg Monthly GB Download": customer_dict["Avg_Monthly_GB_Download"],
        "Online Security": customer_dict["Online_Security"],
        "Online Backup": customer_dict["Online_Backup"],
        "Device Protection Plan": customer_dict["Device_Protection_Plan"],
        "Premium Tech Support": customer_dict["Premium_Tech_Support"],
        "Streaming TV": customer_dict["Streaming_TV"],
        "Streaming Movies": customer_dict["Streaming_Movies"],
        "Streaming Music": customer_dict["Streaming_Music"],
        "Unlimited Data": customer_dict["Unlimited_Data"],
        "Contract": customer_dict["Contract"],
        "Paperless Billing": customer_dict["Paperless_Billing"],
        "Payment Method": customer_dict["Payment_Method"],
        "Monthly Charge": customer_dict["Monthly_Charge"],
        "Total Charges": customer_dict["Total_Charges"],
        "Total Refunds": customer_dict["Total_Refunds"],
        "Total Extra Data Charges": customer_dict["Total_Extra_Data_Charges"],
        "Total Long Distance Charges": customer_dict["Total_Long_Distance_Charges"],
        "Total Revenue": customer_dict["Total_Revenue"],
    }


@router.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        customer_dict = customer.model_dump()
        formatted = _format_customer(customer_dict)

        prediction_result = predict_churn(formatted)
        probability = prediction_result["churn_probability"]
        risk_level = prediction_result["risk_level"]

        ltv_revenue_at_risk = calculate_ltv_revenue_at_risk(
            probability=probability,
            monthly_charge=formatted["Monthly Charge"],
            tenure=formatted["Tenure in Months"],
            churn_rate=churn_rate,
        )

        recommended_action = recommend_action(
            probability=probability,
            contract=formatted["Contract"],
            tenure=formatted["Tenure in Months"],
            tech_support=formatted["Premium Tech Support"],
        )

        # Expected revenue loss = churn_prob × Total Revenue
        expected_revenue_loss = round(probability * formatted["Total Revenue"], 2)

        # SHAP explanation (non-blocking)
        shap_explanation = None
        try:
            from services.shap_explainer import explain_prediction
            shap_explanation = explain_prediction(formatted)
        except Exception as shap_err:
            logger.warning(f"SHAP explanation failed: {shap_err}")

        # Customer segment
        segment = predict_segment(formatted)

        return {
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "ltv_revenue_at_risk": round(ltv_revenue_at_risk, 2),
            "recommended_action": recommended_action,
            "expected_revenue_loss": expected_revenue_loss,
            "shap_explanation": shap_explanation,
            "segment": segment,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
