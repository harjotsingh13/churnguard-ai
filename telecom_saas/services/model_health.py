"""
Model Health Monitoring Service
Reports model performance metrics and drift status.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def get_model_health():
    """Return model performance metrics and health indicators."""
    logger.info("Fetching model health metrics...")

    # Load comparison results for metrics
    comparison_path = ARTIFACTS_DIR / "model_comparison.json"
    if not comparison_path.exists():
        return None

    with open(comparison_path) as f:
        comparison = json.load(f)

    best_model_name = comparison.get("best_model", "Unknown")
    best_metrics = comparison.get("models", {}).get(best_model_name, {})

    # Get last trained timestamp from model file
    model_path = ARTIFACTS_DIR / "churn_model.pkl"
    if model_path.exists():
        mtime = os.path.getmtime(model_path)
        last_trained = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    else:
        last_trained = "Unknown"

    # Quick drift check
    drift_detected = False
    drift_details = None
    try:
        from services.data_drift import check_drift
        drift_result = check_drift()
        drift_detected = drift_result["drift_detected"]
        if drift_detected:
            drift_details = drift_result["summary"]
    except Exception as e:
        logger.warning(f"Could not check drift: {e}")

    return {
        "model_name": best_model_name,
        "auc": best_metrics.get("roc_auc", 0),
        "accuracy": best_metrics.get("accuracy", 0),
        "precision": best_metrics.get("precision", 0),
        "recall": best_metrics.get("recall", 0),
        "f1": best_metrics.get("f1", 0),
        "last_trained": last_trained,
        "drift_detected": drift_detected,
        "drift_details": drift_details,
    }
