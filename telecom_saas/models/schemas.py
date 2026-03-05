"""
Pydantic request/response schemas for the ChurnGuard AI API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


# --------------------------------------------------
# Request Schemas
# --------------------------------------------------

class CustomerData(BaseModel):
    Gender: str
    Age: int
    Married: str
    Number_of_Dependents: int
    Number_of_Referrals: int
    Tenure_in_Months: int
    Offer: str
    Phone_Service: str
    Avg_Monthly_Long_Distance_Charges: float
    Multiple_Lines: str
    Internet_Service: str
    Internet_Type: str
    Avg_Monthly_GB_Download: float
    Online_Security: str
    Online_Backup: str
    Device_Protection_Plan: str
    Premium_Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Streaming_Music: str
    Unlimited_Data: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
    Monthly_Charge: float
    Total_Charges: float
    Total_Refunds: float
    Total_Extra_Data_Charges: float
    Total_Long_Distance_Charges: float
    Total_Revenue: float


# --------------------------------------------------
# Response Schemas
# --------------------------------------------------

class PredictionResponse(BaseModel):
    churn_probability: float
    risk_level: str
    ltv_revenue_at_risk: float
    recommended_action: str
    expected_revenue_loss: float = 0.0
    shap_explanation: Optional[Dict] = None
    segment: Optional[Dict] = None


class RiskBucket(BaseModel):
    bucket: str
    count: int
    percentage: float
    revenue_at_risk: float


class BatchPredictionResponse(BaseModel):
    total_customers: int
    total_revenue_at_risk: float
    high_risk_percentage: float
    medium_risk_percentage: float
    low_risk_percentage: float
    risk_distribution: List[RiskBucket] = []
    top_10_high_value_risky_customers: List[Dict]


class SimulationResponse(BaseModel):
    original_revenue_at_risk: float
    revenue_after_strategy: float
    revenue_saved: float


class CampaignResponse(BaseModel):
    total_customers_scored: int
    customers_selected: int
    total_budget: float
    campaign_cost: float
    total_revenue_protected: float
    net_roi: float
    roi_percentage: float
    selected_customers: List[Dict]


class ModelHealthResponse(BaseModel):
    model_name: str
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    last_trained: str
    drift_detected: bool
    drift_details: Optional[str] = None


class DriftFeature(BaseModel):
    feature: str
    psi: float
    status: str


class DriftResponse(BaseModel):
    drift_detected: bool
    overall_psi: float
    threshold: float
    features: List[DriftFeature]
    summary: str


class TrendPoint(BaseModel):
    month: str
    churn_rate: float
    revenue_at_risk: float
    churned_count: int


class TrendResponse(BaseModel):
    trends: List[TrendPoint]


class GeoBucket(BaseModel):
    city: str
    total_customers: int
    churned: int
    churn_rate: float
    revenue_at_risk: float


class GeographicResponse(BaseModel):
    regions: List[GeoBucket]
