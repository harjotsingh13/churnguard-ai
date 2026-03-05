"""
Geographic Analysis Service
Computes churn and revenue metrics grouped by city.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR.parent / "telecom_customer_churn.csv"


def compute_geographic():
    """Compute churn rate and revenue at risk by top cities."""
    logger.info("Computing geographic analytics...")

    df = pd.read_csv(DATA_PATH)
    df = df[df["Customer Status"] != "Joined"].copy()
    df["Churn"] = (df["Customer Status"] == "Churned").astype(int)

    if "City" not in df.columns:
        logger.warning("No City column found, returning empty result")
        return {"regions": []}

    # Group by city
    grouped = df.groupby("City").agg(
        total_customers=("Churn", "count"),
        churned=("Churn", "sum"),
        total_revenue=("Total Revenue", "sum"),
        churned_revenue=("Total Revenue", lambda x: x[df.loc[x.index, "Churn"] == 1].sum()),
    ).reset_index()

    grouped["churn_rate"] = round(grouped["churned"] / grouped["total_customers"] * 100, 1)
    grouped["revenue_at_risk"] = round(grouped["churned_revenue"], 2)

    # Take top 15 cities by customer count
    top = grouped.nlargest(15, "total_customers")

    regions = []
    for _, row in top.iterrows():
        regions.append({
            "city": str(row["City"]),
            "total_customers": int(row["total_customers"]),
            "churned": int(row["churned"]),
            "churn_rate": float(row["churn_rate"]),
            "revenue_at_risk": float(row["revenue_at_risk"]),
        })

    logger.info(f"Geographic analytics: {len(regions)} cities analyzed")
    return {"regions": regions}
