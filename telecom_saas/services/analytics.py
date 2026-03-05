"""
Analytics Module
Precomputes KPI metrics and churn analytics from the training dataset.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR.parent / "telecom_customer_churn.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def compute_analytics():
    """Compute comprehensive churn analytics from the dataset."""
    logger.info("Computing analytics from training data...")
    
    df = pd.read_csv(DATA_PATH)
    df_active = df[df["Customer Status"] != "Joined"].copy()
    df_active["Churn"] = (df_active["Customer Status"] == "Churned").astype(int)
    
    # 1. Overview KPIs
    total_customers = len(df_active)
    churned = df_active["Churn"].sum()
    churn_rate = round(churned / total_customers * 100, 1)
    avg_monthly_charge = round(df_active["Monthly Charge"].mean(), 2)
    avg_tenure = round(df_active["Tenure in Months"].mean(), 1)
    total_revenue = round(df_active["Total Revenue"].sum(), 2)
    avg_clv = round(df_active["Total Revenue"].mean(), 2)
    revenue_at_risk = round(df_active[df_active["Churn"] == 1]["Total Revenue"].sum(), 2)
    
    overview = {
        "total_customers": int(total_customers),
        "churned_customers": int(churned),
        "churn_rate": churn_rate,
        "avg_monthly_charge": avg_monthly_charge,
        "avg_tenure_months": avg_tenure,
        "total_revenue": total_revenue,
        "avg_clv": avg_clv,
        "revenue_at_risk": revenue_at_risk,
    }
    
    # 2. Churn by Contract Type
    churn_by_contract = []
    for contract_type in ["Month-to-month", "One Year", "Two Year"]:
        subset = df_active[df_active["Contract"] == contract_type]
        if len(subset) > 0:
            churn_by_contract.append({
                "category": contract_type,
                "total": int(len(subset)),
                "churned": int(subset["Churn"].sum()),
                "churn_rate": round(subset["Churn"].mean() * 100, 1)
            })
    
    # 3. Churn by Internet Type
    churn_by_internet = []
    for itype in df_active["Internet Type"].dropna().unique():
        subset = df_active[df_active["Internet Type"] == itype]
        if len(subset) > 0:
            churn_by_internet.append({
                "category": str(itype),
                "total": int(len(subset)),
                "churned": int(subset["Churn"].sum()),
                "churn_rate": round(subset["Churn"].mean() * 100, 1)
            })
    churn_by_internet.sort(key=lambda x: x["churn_rate"], reverse=True)
    
    # 4. Churn by Payment Method
    churn_by_payment = []
    for method in df_active["Payment Method"].dropna().unique():
        subset = df_active[df_active["Payment Method"] == method]
        if len(subset) > 0:
            churn_by_payment.append({
                "category": str(method),
                "total": int(len(subset)),
                "churned": int(subset["Churn"].sum()),
                "churn_rate": round(subset["Churn"].mean() * 100, 1)
            })
    churn_by_payment.sort(key=lambda x: x["churn_rate"], reverse=True)
    
    # 5. Top Churn Reasons
    churn_reasons = []
    if "Churn Reason" in df.columns:
        reasons = df[df["Customer Status"] == "Churned"]["Churn Reason"].value_counts().head(10)
        for reason, count in reasons.items():
            churn_reasons.append({
                "reason": str(reason),
                "count": int(count),
                "percentage": round(count / churned * 100, 1)
            })
    
    # 6. Churn by Tenure Buckets
    bins = [0, 6, 12, 24, 48, 100]
    labels = ["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49+ mo"]
    df_active["tenure_bucket"] = pd.cut(df_active["Tenure in Months"], bins=bins, labels=labels)
    churn_by_tenure = []
    for bucket in labels:
        subset = df_active[df_active["tenure_bucket"] == bucket]
        if len(subset) > 0:
            churn_by_tenure.append({
                "category": bucket,
                "total": int(len(subset)),
                "churned": int(subset["Churn"].sum()),
                "churn_rate": round(subset["Churn"].mean() * 100, 1)
            })
    
    analytics = {
        "overview": overview,
        "churn_by_contract": churn_by_contract,
        "churn_by_internet": churn_by_internet,
        "churn_by_payment": churn_by_payment,
        "churn_by_tenure": churn_by_tenure,
        "top_churn_reasons": churn_reasons,
    }
    
    # Save to artifacts
    with open(ARTIFACTS_DIR / "analytics.json", "w") as f:
        json.dump(analytics, f, indent=2)
    
    logger.info(f"Analytics computed: {total_customers} customers, {churn_rate}% churn rate")
    return analytics


def load_analytics():
    """Load precomputed analytics."""
    path = ARTIFACTS_DIR / "analytics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Compute if not cached
    return compute_analytics()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    result = compute_analytics()
    print(json.dumps(result["overview"], indent=2))
    print(f"\nTop 5 churn reasons:")
    for r in result["top_churn_reasons"][:5]:
        print(f"  {r['reason']}: {r['count']} ({r['percentage']}%)")
