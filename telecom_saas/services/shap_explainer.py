"""
SHAP Explainability Module
Provides model-agnostic explanations for churn predictions using SHAP values.
"""

import base64
import io
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Load model and features
_model = joblib.load(ARTIFACTS_DIR / "churn_model.pkl")
_features = joblib.load(ARTIFACTS_DIR / "model_features.pkl")

# Separate preprocessor and classifier from pipeline
_preprocessor = _model.named_steps["preprocessor"]
_classifier = _model.named_steps["classifier"]

# Build SHAP explainer
_explainer = None
_global_shap_values = None
_global_feature_importance = None


def _get_transformed_feature_names():
    """Get feature names after preprocessing (one-hot encoding expands categoricals)."""
    try:
        return _preprocessor.get_feature_names_out()
    except Exception:
        return None


def _init_explainer():
    """Initialize SHAP explainer lazily (first call only)."""
    global _explainer
    if _explainer is None:
        logger.info("Initializing SHAP TreeExplainer...")
        _explainer = shap.TreeExplainer(_classifier)
    return _explainer


def explain_prediction(customer_data: dict):
    """
    Generate SHAP explanation for a single customer prediction.
    
    Returns dict with:
        - top_drivers: features pushing toward churn
        - top_protectors: features reducing churn risk
        - explanation_text: natural language summary
        - shap_plot_base64: waterfall chart as base64 PNG
    """
    explainer = _init_explainer()
    
    # Clean NaN values
    from services.predict import _clean_nan
    customer_data = _clean_nan(customer_data)
    
    # Create DataFrame and transform
    input_df = pd.DataFrame([customer_data])
    X_transformed = _preprocessor.transform(input_df)
    
    # Get SHAP values
    shap_values = explainer.shap_values(X_transformed)
    
    # For binary classification, use class 1 (churn) values
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Class 1 (churn), first sample
    elif len(shap_values.shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]
    
    # Map SHAP values back to original feature names
    transformed_names = _get_transformed_feature_names()
    
    if transformed_names is not None and len(transformed_names) == len(sv):
        # Aggregate one-hot encoded features back to original feature names
        feature_shap = _aggregate_shap_to_original(sv, transformed_names, customer_data)
    else:
        # Fallback: use original feature names directly
        feature_shap = {f: float(sv[i]) for i, f in enumerate(_features) if i < len(sv)}
    
    # Sort by absolute impact
    sorted_features = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Top drivers (positive SHAP = pushes toward churn)
    top_drivers = [
        {"feature": f, "impact": round(v, 4), "value": str(customer_data.get(f, "N/A"))}
        for f, v in sorted_features if v > 0
    ][:5]
    
    # Top protectors (negative SHAP = reduces churn)
    top_protectors = [
        {"feature": f, "impact": round(v, 4), "value": str(customer_data.get(f, "N/A"))}
        for f, v in sorted_features if v < 0
    ][:5]
    
    # Generate natural language explanation
    explanation_text = _generate_explanation_text(top_drivers, top_protectors)
    
    # Generate SHAP bar chart
    shap_plot_b64 = _generate_shap_bar_chart(sorted_features[:10])
    
    return {
        "top_drivers": top_drivers,
        "top_protectors": top_protectors,
        "explanation_text": explanation_text,
        "shap_plot_base64": shap_plot_b64
    }


def _aggregate_shap_to_original(shap_values, transformed_names, customer_data):
    """Aggregate SHAP values from one-hot encoded features back to original feature names."""
    feature_shap = {}
    
    for i, tname in enumerate(transformed_names):
        tname = str(tname)
        original_feature = None
        
        # Match transformed name to original feature
        for f in _features:
            # ColumnTransformer names: "num__FeatureName" or "cat__FeatureName_Value"
            f_underscore = f.replace(" ", "_")
            if tname == f"num__{f}" or tname == f"num__{f_underscore}":
                original_feature = f
                break
            if tname.startswith(f"cat__{f}_") or tname.startswith(f"cat__{f_underscore}_"):
                original_feature = f
                break
        
        if original_feature:
            feature_shap[original_feature] = feature_shap.get(original_feature, 0) + float(shap_values[i])
        else:
            feature_shap[tname] = float(shap_values[i])
    
    return feature_shap


def _generate_explanation_text(drivers, protectors):
    """Generate a human-readable explanation of the prediction."""
    parts = []
    
    if drivers:
        driver_parts = [f"**{d['feature']}** ({d['value']}, +{abs(d['impact']):.3f})" for d in drivers[:3]]
        parts.append(f"This customer's churn risk is driven primarily by {', '.join(driver_parts)}.")
    
    if protectors:
        protector_parts = [f"**{p['feature']}** ({p['value']}, {p['impact']:.3f})" for p in protectors[:3]]
        parts.append(f"Factors reducing risk include {', '.join(protector_parts)}.")
    
    if not parts:
        return "No significant factors identified for this prediction."
    
    return " ".join(parts)


def _generate_shap_bar_chart(sorted_features):
    """Generate a horizontal bar chart of SHAP values, return as base64 PNG."""
    features = [f[0] for f in reversed(sorted_features)]
    values = [f[1] for f in reversed(sorted_features)]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]
    bars = ax.barh(features, values, color=colors, height=0.6, edgecolor="none")
    
    ax.axvline(x=0, color="#888", linewidth=0.8, linestyle="-")
    ax.set_xlabel("SHAP Value (Impact on Churn Prediction)", fontsize=10)
    ax.set_title("Feature Impact Analysis", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = bar.get_width() + (0.005 if val >= 0 else -0.005)
        ha = "left" if val >= 0 else "right"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                va="center", ha=ha, fontsize=8, color="#333")
    
    fig.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.read()).decode("utf-8")


def compute_global_feature_importance():
    """Compute global SHAP feature importance using a sample of the training data."""
    global _global_feature_importance
    
    if _global_feature_importance is not None:
        return _global_feature_importance
    
    logger.info("Computing global feature importance (this may take a moment)...")
    
    from services.predict import _clean_nan
    
    # Load a sample of data
    df = pd.read_csv(BASE_DIR.parent / "telecom_customer_churn.csv")
    df = df[df["Customer Status"] != "Joined"]
    features = joblib.load(ARTIFACTS_DIR / "model_features.pkl")
    X = df[features].sample(min(200, len(df)), random_state=42)
    
    # Clean NaN
    string_cols = X.select_dtypes(include="object").columns
    numeric_cols = X.select_dtypes(include="number").columns
    X[string_cols] = X[string_cols].fillna("No")
    X[numeric_cols] = X[numeric_cols].fillna(0)
    
    # Transform and compute SHAP
    X_transformed = _preprocessor.transform(X)
    explainer = _init_explainer()
    shap_values = explainer.shap_values(X_transformed)
    
    if isinstance(shap_values, list):
        sv = shap_values[1]  # Class 1 (churn)
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1] # Class 1 (churn)
    else:
        sv = shap_values
    
    # Mean absolute SHAP values per feature
    transformed_names = _get_transformed_feature_names()
    mean_abs_shap = np.abs(sv).mean(axis=0)
    
    # Aggregate to original features
    feature_importance = {}
    if transformed_names is not None and len(transformed_names) == len(mean_abs_shap):
        for i, tname in enumerate(transformed_names):
            tname = str(tname)
            original = None
            for f in features:
                f_underscore = f.replace(" ", "_")
                if tname == f"num__{f}" or tname == f"num__{f_underscore}":
                    original = f
                    break
                if tname.startswith(f"cat__{f}_") or tname.startswith(f"cat__{f_underscore}_"):
                    original = f
                    break
            if original:
                feature_importance[original] = feature_importance.get(original, 0) + float(mean_abs_shap[i])
    else:
        for i, f in enumerate(features):
            if i < len(mean_abs_shap):
                feature_importance[f] = float(mean_abs_shap[i])
    
    # Sort by importance
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    _global_feature_importance = [
        {"feature": f, "importance": round(float(v), 4)}
        for f, v in sorted_importance
    ]
    
    logger.info("Global feature importance computed successfully.")
    return _global_feature_importance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test single prediction explanation
    sample = {
        "Gender": "Male", "Age": 35, "Married": "Yes", "Number of Dependents": 0,
        "Number of Referrals": 1, "Tenure in Months": 4, "Offer": "None",
        "Phone Service": "Yes", "Avg Monthly Long Distance Charges": 10,
        "Multiple Lines": "No", "Internet Service": "Yes", "Internet Type": "Fiber Optic",
        "Avg Monthly GB Download": 50, "Online Security": "No", "Online Backup": "No",
        "Device Protection Plan": "No", "Premium Tech Support": "No",
        "Streaming TV": "Yes", "Streaming Movies": "Yes", "Streaming Music": "No",
        "Unlimited Data": "Yes", "Contract": "Month-to-month",
        "Paperless Billing": "Yes", "Payment Method": "Bank Withdrawal",
        "Monthly Charge": 80, "Total Charges": 320, "Total Refunds": 0,
        "Total Extra Data Charges": 0, "Total Long Distance Charges": 40, "Total Revenue": 360
    }
    
    result = explain_prediction(sample)
    print("Top Drivers:", result["top_drivers"])
    print("Top Protectors:", result["top_protectors"])
    print("Explanation:", result["explanation_text"])
    print("Plot generated:", len(result["shap_plot_base64"]) > 0)
    
    # Test global importance
    gi = compute_global_feature_importance()
    print("\nGlobal top 5:", gi[:5])
