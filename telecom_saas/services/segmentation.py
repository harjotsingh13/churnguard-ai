"""
Customer Segmentation Module
Groups customers into business-meaningful segments using K-Means clustering.
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR.parent / "telecom_customer_churn.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Segment labels and retention strategies
SEGMENT_PROFILES = {
    0: {
        "name": "New Price-Sensitive",
        "description": "Short-tenure customers with lower spending, often on month-to-month contracts",
        "retention_strategy": "Offer loyalty discounts and contract upgrade incentives to build commitment early"
    },
    1: {
        "name": "Loyal Premium",
        "description": "Long-tenure, high-spending customers with stable contracts and multiple services",
        "retention_strategy": "Provide VIP perks, priority support, and exclusive early-access offers to maintain loyalty"
    },
    2: {
        "name": "At-Risk Mid-Tier",
        "description": "Medium-tenure customers showing signs of disengagement with moderate spending",
        "retention_strategy": "Proactive outreach with personalized bundles and satisfaction surveys"
    },
    3: {
        "name": "Digital Streamers",
        "description": "Customers with high data usage and streaming services, tech-savvy profile",
        "retention_strategy": "Enhance digital experience, offer streaming partnerships and data boost packages"
    }
}


def load_and_prepare_data():
    """Load dataset and prepare features for clustering."""
    df = pd.read_csv(DATA_PATH)
    features = joblib.load(ARTIFACTS_DIR / "model_features.pkl")
    
    # Drop 'Joined' status
    df = df[df["Customer Status"] != "Joined"].copy()
    df["Churn"] = (df["Customer Status"] == "Churned").astype(int)
    
    # Select clustering features (key behavioral + financial features)
    cluster_features = [
        "Tenure in Months",
        "Monthly Charge",
        "Total Revenue",
        "Total Charges",
        "Number of Referrals",
        "Avg Monthly GB Download",
    ]
    
    # Count number of services
    service_cols = [
        "Phone Service", "Internet Service", "Online Security",
        "Online Backup", "Device Protection Plan", "Premium Tech Support",
        "Streaming TV", "Streaming Movies", "Streaming Music"
    ]
    
    # Fill NaN
    for col in cluster_features:
        df[col] = df[col].fillna(0)
    
    # Count services
    df["num_services"] = 0
    for col in service_cols:
        df["num_services"] += (df[col] == "Yes").astype(int)
    
    cluster_features.append("num_services")
    
    # Encode contract to numeric
    contract_map = {"Month-to-month": 0, "One Year": 1, "Two Year": 2}
    df["contract_num"] = df["Contract"].map(contract_map).fillna(0)
    cluster_features.append("contract_num")
    
    return df, cluster_features


def find_optimal_k(X_scaled, max_k=8):
    """Use elbow method to find optimal K."""
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    # Simple elbow detection: find the point of maximum curvature
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    optimal_k = np.argmax(np.abs(diffs2)) + 2  # +2 because of double diff offset
    optimal_k = max(3, min(optimal_k, 5))  # Clamp between 3-5
    
    return optimal_k


def run_segmentation():
    """Run K-Means clustering and generate segment profiles."""
    logger.info("Loading and preparing data for segmentation...")
    df, cluster_features = load_and_prepare_data()
    
    X = df[cluster_features].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal K (or use 4 which maps well to our predefined profiles)
    n_clusters = 4
    logger.info(f"Running K-Means with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["segment_id"] = kmeans.fit_predict(X_scaled)
    
    # Compute segment statistics
    segment_stats = []
    for seg_id in range(n_clusters):
        seg_df = df[df["segment_id"] == seg_id]
        
        # Determine the best matching profile based on characteristics
        stats = {
            "segment_id": seg_id,
            "customer_count": int(len(seg_df)),
            "pct_of_total": round(len(seg_df) / len(df) * 100, 1),
            "avg_tenure": round(float(seg_df["Tenure in Months"].mean()), 1),
            "avg_monthly_charge": round(float(seg_df["Monthly Charge"].mean()), 2),
            "avg_total_revenue": round(float(seg_df["Total Revenue"].mean()), 2),
            "avg_services": round(float(seg_df["num_services"].mean()), 1),
            "churn_rate": round(float(seg_df["Churn"].mean() * 100), 1),
            "dominant_contract": seg_df["Contract"].mode().iloc[0] if len(seg_df) > 0 else "Unknown",
            "total_revenue_at_risk": round(float(
                seg_df[seg_df["Churn"] == 1]["Total Revenue"].sum()
            ), 2)
        }
        segment_stats.append(stats)
    
    # Sort segments by churn rate to assign meaningful labels
    # Highest churn = most at-risk, lowest churn + high revenue = loyal premium
    sorted_by_churn = sorted(range(n_clusters), key=lambda i: segment_stats[i]["churn_rate"], reverse=True)
    sorted_by_revenue = sorted(range(n_clusters), key=lambda i: segment_stats[i]["avg_total_revenue"], reverse=True)
    
    # Assign profiles intelligently
    label_mapping = {}
    used_profiles = set()
    
    # Highest churn rate + low tenure = "New Price-Sensitive" or "At-Risk Mid-Tier"
    for seg_id in sorted_by_churn:
        stats = segment_stats[seg_id]
        if 0 not in used_profiles and stats["avg_tenure"] < df["Tenure in Months"].median():
            label_mapping[seg_id] = 0  # New Price-Sensitive
            used_profiles.add(0)
        elif 2 not in used_profiles and stats["churn_rate"] > df["Churn"].mean() * 100:
            label_mapping[seg_id] = 2  # At-Risk Mid-Tier
            used_profiles.add(2)
        elif 1 not in used_profiles and stats["avg_total_revenue"] > df["Total Revenue"].mean():
            label_mapping[seg_id] = 1  # Loyal Premium
            used_profiles.add(1)
        elif 3 not in used_profiles:
            label_mapping[seg_id] = 3  # Digital Streamers
            used_profiles.add(3)
    
    # Fill any remaining
    for seg_id in range(n_clusters):
        if seg_id not in label_mapping:
            for p in range(4):
                if p not in used_profiles:
                    label_mapping[seg_id] = p
                    used_profiles.add(p)
                    break
    
    # Apply labels to stats
    for stats in segment_stats:
        profile_id = label_mapping.get(stats["segment_id"], 0)
        profile = SEGMENT_PROFILES[profile_id]
        stats["name"] = profile["name"]
        stats["description"] = profile["description"]
        stats["retention_strategy"] = profile["retention_strategy"]
    
    # Save artifacts
    output = {
        "n_clusters": n_clusters,
        "segments": segment_stats,
        "cluster_features": cluster_features,
        "total_customers": len(df),
    }
    
    with open(ARTIFACTS_DIR / "segment_profiles.json", "w") as f:
        json.dump(output, f, indent=2)
    
    # Save models for real-time segment prediction
    joblib.dump(kmeans, ARTIFACTS_DIR / "segmentation_model.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "segmentation_scaler.pkl")
    
    # Save mapping
    with open(ARTIFACTS_DIR / "segment_mapping.json", "w") as f:
        json.dump({
            "label_mapping": {str(k): v for k, v in label_mapping.items()},
            "cluster_features": cluster_features,
            "profiles": {str(k): v for k, v in SEGMENT_PROFILES.items()}
        }, f, indent=2)
    
    logger.info("Segmentation complete!")
    for s in segment_stats:
        logger.info(f"  {s['name']}: {s['customer_count']} customers, {s['churn_rate']}% churn rate")
    
    return output


def predict_segment(customer_data: dict):
    """Predict which segment a customer belongs to."""
    try:
        kmeans = joblib.load(ARTIFACTS_DIR / "segmentation_model.pkl")
        scaler = joblib.load(ARTIFACTS_DIR / "segmentation_scaler.pkl")
        mapping_data = json.load(open(ARTIFACTS_DIR / "segment_mapping.json"))
        
        label_mapping = {int(k): v for k, v in mapping_data["label_mapping"].items()}
        cluster_features = mapping_data["cluster_features"]
        
        # Extract features
        feature_values = []
        for feat in cluster_features:
            if feat == "num_services":
                # Count services
                service_cols = [
                    "Phone Service", "Internet Service", "Online Security",
                    "Online Backup", "Device Protection Plan", "Premium Tech Support",
                    "Streaming TV", "Streaming Movies", "Streaming Music"
                ]
                count = sum(1 for c in service_cols if customer_data.get(c, "No") == "Yes")
                feature_values.append(count)
            elif feat == "contract_num":
                contract_map = {"Month-to-month": 0, "Month-to-Month": 0, "One Year": 1, "Two Year": 2}
                feature_values.append(contract_map.get(customer_data.get("Contract", "Month-to-month"), 0))
            else:
                val = customer_data.get(feat, 0)
                feature_values.append(float(val) if val is not None and val == val else 0)
        
        X = np.array([feature_values])
        X_scaled = scaler.transform(X)
        cluster_id = int(kmeans.predict(X_scaled)[0])
        
        profile_id = label_mapping.get(cluster_id, 0)
        profile = SEGMENT_PROFILES[profile_id]
        
        return {
            "segment_id": cluster_id,
            "segment_name": profile["name"],
            "segment_description": profile["description"],
            "retention_strategy": profile["retention_strategy"]
        }
    except Exception as e:
        logger.warning(f"Segment prediction failed: {e}")
        return {
            "segment_id": -1,
            "segment_name": "Unknown",
            "segment_description": "Segmentation model not available",
            "retention_strategy": "Contact customer for more information"
        }


def load_segment_profiles():
    """Load precomputed segment profiles."""
    path = ARTIFACTS_DIR / "segment_profiles.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_segmentation()
