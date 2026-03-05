import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load trained pipeline
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "artifacts" / "churn_model.pkl")

# String feature names (categorical columns that must not contain float NaN)
_STRING_FEATURES = {
    "Gender", "Married", "Offer", "Phone Service", "Multiple Lines",
    "Internet Service", "Internet Type", "Online Security", "Online Backup",
    "Device Protection Plan", "Premium Tech Support", "Streaming TV",
    "Streaming Movies", "Streaming Music", "Unlimited Data", "Contract",
    "Paperless Billing", "Payment Method"
}


def get_risk_level(probability):
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


def _clean_nan(data: dict) -> dict:
    """Replace NaN values with appropriate defaults."""
    cleaned = {}
    for k, v in data.items():
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cleaned[k] = "No" if k in _STRING_FEATURES else 0
            else:
                cleaned[k] = v
        except (TypeError, ValueError):
            cleaned[k] = v
    return cleaned


def predict_churn(customer_data: dict):
    """
    Predict churn probability and revenue at risk.
    """

    # Clean NaN values from input
    customer_data = _clean_nan(customer_data)

    # Convert input to DataFrame
    input_df = pd.DataFrame([customer_data])

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    # Risk segmentation
    risk_level = get_risk_level(probability)

    # Revenue at Risk calculation
    monthly_charge = customer_data.get("Monthly Charge", 0)
    revenue_at_risk = probability * monthly_charge

    return {
    "churn_probability": float(round(probability, 4)),
    "risk_level": risk_level,
    "revenue_at_risk": float(round(revenue_at_risk, 2))
    }


# Example usage
if __name__ == "__main__":
    sample_customer = {
        "Gender": "Male",
        "Age": 35,
        "Married": "Yes",
        "Number of Dependents": 0,
        "Number of Referrals": 1,
        "Tenure in Months": 12,
        "Offer": "Offer A",
        "Phone Service": "Yes",
        "Avg Monthly Long Distance Charges": 10,
        "Multiple Lines": "No",
        "Internet Service": "Yes",
        "Internet Type": "Fiber Optic",
        "Avg Monthly GB Download": 50,
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection Plan": "No",
        "Premium Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Streaming Music": "No",
        "Unlimited Data": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Bank Withdrawal",
        "Monthly Charge": 80,
        "Total Charges": 960,
        "Total Refunds": 0,
        "Total Extra Data Charges": 0,
        "Total Long Distance Charges": 120,
        "Total Revenue": 1080
    }

    result = predict_churn(sample_customer)
    print(result)
