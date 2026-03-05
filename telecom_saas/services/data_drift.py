"""
Data Drift Detection Service
Uses Population Stability Index (PSI) to detect distribution shifts.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR.parent / "telecom_customer_churn.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

PSI_THRESHOLD = 0.2  # PSI > 0.2 indicates significant drift


def _compute_psi(expected, actual, bins=10):
    """Compute Population Stability Index between two distributions."""
    # Create bins from expected distribution
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )

    expected_counts = np.histogram(expected, bins=breakpoints)[0] + 1
    actual_counts = np.histogram(actual, bins=breakpoints)[0] + 1

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def check_drift():
    """
    Compare training data distribution vs a simulated 'current' batch.

    Since we don't have separate production data, we simulate by splitting
    the dataset: first 80% = training reference, last 20% = current batch.
    """
    logger.info("Running data drift detection...")

    df = pd.read_csv(DATA_PATH)
    df = df[df["Customer Status"] != "Joined"].copy()

    features = joblib.load(ARTIFACTS_DIR / "model_features.pkl")

    # Select numeric features for drift analysis
    numeric_features = []
    for f in features:
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
            numeric_features.append(f)

    # Split: 80% reference, 20% simulated current
    split_idx = int(len(df) * 0.8)
    df_ref = df.iloc[:split_idx]
    df_curr = df.iloc[split_idx:]

    feature_results = []
    overall_psi = 0.0

    for feat in numeric_features:
        ref_vals = df_ref[feat].dropna().values
        curr_vals = df_curr[feat].dropna().values

        if len(ref_vals) < 10 or len(curr_vals) < 10:
            continue

        psi = _compute_psi(ref_vals, curr_vals)
        overall_psi += psi

        status = "stable"
        if psi > PSI_THRESHOLD:
            status = "drift"
        elif psi > 0.1:
            status = "warning"

        feature_results.append({
            "feature": feat,
            "psi": psi,
            "status": status,
        })

    # Sort by PSI descending
    feature_results.sort(key=lambda x: x["psi"], reverse=True)

    n_features = len(feature_results)
    avg_psi = round(overall_psi / max(n_features, 1), 4)
    drift_detected = avg_psi > PSI_THRESHOLD

    drifted = [f for f in feature_results if f["status"] == "drift"]

    if drift_detected:
        summary = f"Drift detected in {len(drifted)}/{n_features} features. Average PSI: {avg_psi}"
    else:
        summary = f"No significant drift. Average PSI: {avg_psi} (threshold: {PSI_THRESHOLD})"

    logger.info(summary)

    return {
        "drift_detected": drift_detected,
        "overall_psi": avg_psi,
        "threshold": PSI_THRESHOLD,
        "features": feature_results,
        "summary": summary,
    }
