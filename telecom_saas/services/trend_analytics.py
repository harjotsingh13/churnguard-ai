"""
Trend Analytics Service
Simulates monthly churn trends using tenure-based grouping.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR.parent / "telecom_customer_churn.csv"


def compute_trends():
    """Compute simulated monthly churn trends from the dataset."""
    logger.info("Computing trend analytics...")

    df = pd.read_csv(DATA_PATH)
    df = df[df["Customer Status"] != "Joined"].copy()
    df["Churn"] = (df["Customer Status"] == "Churned").astype(int)

    # Simulate 6 months by distributing customers based on tenure
    # Newer customers → more recent months, longer tenure → older months
    df_sorted = df.sort_values("Tenure in Months", ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    chunk = n // 6

    months = ["Oct 2025", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026"]
    trends = []

    for i, month in enumerate(months):
        start = i * chunk
        end = (i + 1) * chunk if i < 5 else n
        subset = df_sorted.iloc[start:end]

        churn_rate = round(float(subset["Churn"].mean() * 100), 1)
        revenue_at_risk = round(float(subset[subset["Churn"] == 1]["Total Revenue"].sum()), 2)
        churned_count = int(subset["Churn"].sum())

        trends.append({
            "month": month,
            "churn_rate": churn_rate,
            "revenue_at_risk": revenue_at_risk,
            "churned_count": churned_count,
        })

    logger.info(f"Trend analytics computed: {len(trends)} months")
    return {"trends": trends}
