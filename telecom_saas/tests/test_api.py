"""
Tests for ChurnGuard AI API endpoints.
Uses FastAPI TestClient for integration testing.
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# --------------------------------------------------
# Sample customer payload for /predict
# --------------------------------------------------

SAMPLE_CUSTOMER = {
    "Gender": "Male",
    "Age": 35,
    "Married": "Yes",
    "Number_of_Dependents": 0,
    "Number_of_Referrals": 1,
    "Tenure_in_Months": 12,
    "Offer": "Offer A",
    "Phone_Service": "Yes",
    "Avg_Monthly_Long_Distance_Charges": 10.0,
    "Multiple_Lines": "No",
    "Internet_Service": "Yes",
    "Internet_Type": "Fiber Optic",
    "Avg_Monthly_GB_Download": 50.0,
    "Online_Security": "No",
    "Online_Backup": "Yes",
    "Device_Protection_Plan": "No",
    "Premium_Tech_Support": "No",
    "Streaming_TV": "Yes",
    "Streaming_Movies": "Yes",
    "Streaming_Music": "No",
    "Unlimited_Data": "Yes",
    "Contract": "Month-to-month",
    "Paperless_Billing": "Yes",
    "Payment_Method": "Bank Withdrawal",
    "Monthly_Charge": 80.0,
    "Total_Charges": 960.0,
    "Total_Refunds": 0.0,
    "Total_Extra_Data_Charges": 0.0,
    "Total_Long_Distance_Charges": 120.0,
    "Total_Revenue": 1080.0,
}


# --------------------------------------------------
# Health Check
# --------------------------------------------------

class TestHealthCheck:
    def test_health_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "API is running successfully"
        assert "version" in data


# --------------------------------------------------
# Individual Prediction
# --------------------------------------------------

class TestPrediction:
    def test_predict_returns_200(self):
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
        assert response.status_code == 200

    def test_predict_response_fields(self):
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
        data = response.json()
        assert "churn_probability" in data
        assert "risk_level" in data
        assert "ltv_revenue_at_risk" in data
        assert "recommended_action" in data
        assert "expected_revenue_loss" in data

    def test_predict_probability_range(self):
        """Churn probability must be between 0 and 1."""
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
        data = response.json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_predict_risk_level_valid(self):
        """Risk level must be Low, Medium, or High."""
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
        data = response.json()
        assert data["risk_level"] in ("Low", "Medium", "High")

    def test_predict_includes_shap_explanation(self):
        """Prediction should include SHAP explanation when available."""
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
        data = response.json()
        # SHAP explanation may be None if it fails, but the key should exist
        assert "shap_explanation" in data

    def test_predict_includes_segment(self):
        """Prediction should include customer segment info."""
        response = client.post("/predict", json=SAMPLE_CUSTOMER)
        data = response.json()
        assert "segment" in data

    def test_predict_missing_fields_returns_422(self):
        """Sending incomplete data should return a validation error."""
        incomplete = {"Gender": "Male", "Age": 35}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422


# --------------------------------------------------
# Analytics
# --------------------------------------------------

class TestAnalytics:
    def test_analytics_overview(self):
        response = client.get("/analytics/overview")
        assert response.status_code == 200

    def test_analytics_all(self):
        response = client.get("/analytics/all")
        assert response.status_code == 200


# --------------------------------------------------
# Model Endpoints
# --------------------------------------------------

class TestModelEndpoints:
    def test_model_comparison(self):
        response = client.get("/model_comparison")
        # 200 if comparison has been run, 404 if not
        assert response.status_code in (200, 404)

    def test_model_health(self):
        response = client.get("/model_health")
        assert response.status_code in (200, 404)


# --------------------------------------------------
# Segments
# --------------------------------------------------

class TestSegments:
    def test_get_segments(self):
        response = client.get("/segments")
        assert response.status_code in (200, 404)


# --------------------------------------------------
# Explainability
# --------------------------------------------------

class TestExplainability:
    def test_data_drift(self):
        response = client.get("/data_drift")
        assert response.status_code == 200
        data = response.json()
        assert "drift_detected" in data
        assert "overall_psi" in data
        assert "features" in data
