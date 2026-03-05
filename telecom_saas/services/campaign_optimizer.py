"""
Campaign Optimizer Service
Budget-constrained retention campaign optimizer.
"""

import logging
from pathlib import Path

import pandas as pd
import joblib

from services.predict import predict_churn
from services.business_logic import calculate_ltv_revenue_at_risk

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def optimize_campaign(df: pd.DataFrame, budget: float, discount_per_customer: float):
    """
    Optimize retention campaign with budget constraints.

    Strategy:
    1. Score all customers for churn probability
    2. Calculate expected_loss = churn_prob × LTV for each
    3. Sort by expected_loss descending
    4. Select customers until budget is exhausted
    5. Return selected customers + ROI metrics
    """
    logger.info(f"Running campaign optimization: budget=${budget}, discount=${discount_per_customer}")

    churn_rate = joblib.load(ARTIFACTS_DIR / "churn_rate.pkl")

    # Fill NaN
    string_cols = df.select_dtypes(include="object").columns
    numeric_cols = df.select_dtypes(include="number").columns
    df[string_cols] = df[string_cols].fillna("No")
    df[numeric_cols] = df[numeric_cols].fillna(0)

    scored = []

    for idx, row in df.iterrows():
        customer_dict = row.to_dict()
        result = predict_churn(customer_dict)
        probability = result["churn_probability"]

        ltv = calculate_ltv_revenue_at_risk(
            probability=probability,
            monthly_charge=customer_dict.get("Monthly Charge", 0),
            tenure=customer_dict.get("Tenure in Months", 0),
            churn_rate=churn_rate,
        )

        expected_loss = probability * ltv

        scored.append({
            "index": int(idx),
            "churn_probability": round(probability, 4),
            "risk_level": result["risk_level"],
            "monthly_charge": round(float(customer_dict.get("Monthly Charge", 0)), 2),
            "tenure": int(customer_dict.get("Tenure in Months", 0)),
            "ltv_at_risk": round(ltv, 2),
            "expected_loss": round(expected_loss, 2),
        })

    # Sort by expected loss descending
    scored.sort(key=lambda x: x["expected_loss"], reverse=True)

    # Select customers until budget exhausted
    selected = []
    total_cost = 0.0
    total_revenue_protected = 0.0

    for customer in scored:
        cost = discount_per_customer
        if total_cost + cost > budget:
            break
        selected.append(customer)
        total_cost += cost
        total_revenue_protected += customer["expected_loss"]

    net_roi = total_revenue_protected - total_cost
    roi_pct = round((net_roi / total_cost * 100), 1) if total_cost > 0 else 0.0

    logger.info(f"Campaign optimized: {len(selected)} customers selected, ROI={roi_pct}%")

    return {
        "total_customers_scored": len(scored),
        "customers_selected": len(selected),
        "total_budget": round(budget, 2),
        "campaign_cost": round(total_cost, 2),
        "total_revenue_protected": round(total_revenue_protected, 2),
        "net_roi": round(net_roi, 2),
        "roi_percentage": roi_pct,
        "selected_customers": selected[:50],  # Limit response to top 50
    }
