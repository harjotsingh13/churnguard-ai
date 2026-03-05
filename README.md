# 🛡️ ChurnGuard AI — Customer Intelligence & Revenue Optimization Platform

A production-grade SaaS application that predicts telecom customer churn, quantifies revenue at risk, and delivers actionable retention strategies — all powered by explainable machine learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-✓-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-✓-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)

---

## 📌 Problem Statement

Telecom companies lose **15–25%** of revenue annually due to customer churn. Traditional churn models only answer *"will they churn?"* — but fail to answer:
- *How much revenue is at risk?*
- *Why is this customer likely to churn?*
- *Which customers should we target with limited retention budgets?*

**ChurnGuard AI** solves all three.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Churn Prediction** | Individual & batch predictions via ML pipeline (scikit-learn) |
| **SHAP Explainability** | Per-prediction top churn drivers & protectors, with visual SHAP charts |
| **LTV & Revenue-at-Risk** | Calculates expected revenue loss per customer using Lifetime Value modeling |
| **Model Comparison** | 4-model benchmark (Logistic Regression, Random Forest, XGBoost, LightGBM) with 5-fold stratified CV |
| **Customer Segmentation** | K-Means clustering into business-meaningful segments with retention strategies |
| **Campaign Optimizer** | Budget-constrained retention targeting with ROI metrics |
| **Strategy Simulator** | "What-if" discount scenario analysis on portfolio data |
| **Data Drift Detection** | Population Stability Index (PSI) monitoring for model health |
| **Analytics Dashboard** | KPI overview, trend analytics, geographic churn breakdown |
| **Modern Frontend** | Single-page SaaS dashboard served at `/app` |

---

## 🏗️ Architecture

```
telecom_saas/
├── app.py                    # FastAPI entry point — mounts all routers & middleware
├── train.ipynb               # Jupyter notebook — model training & evaluation
│
├── routes/                   # API endpoint definitions (thin controllers)
│   ├── prediction.py         #   POST /predict
│   ├── portfolio.py          #   POST /batch_predict
│   ├── strategy.py           #   POST /simulate_strategy
│   ├── campaign.py           #   POST /optimize_campaign
│   ├── analytics.py          #   GET  /analytics/overview, /all, /trends, /geographic
│   ├── models.py             #   GET  /model_comparison, /model_health
│   ├── segments.py           #   GET  /segments
│   └── explainability.py     #   GET  /global_feature_importance, /data_drift
│
├── services/                 # Business logic & ML services
│   ├── predict.py            #   Core churn prediction engine
│   ├── business_logic.py     #   LTV calculation & retention action rules
│   ├── shap_explainer.py     #   SHAP-based model explanation (local + global)
│   ├── model_comparison.py   #   4-model training & comparison pipeline
│   ├── segmentation.py       #   K-Means customer segmentation
│   ├── campaign_optimizer.py #   Budget-constrained campaign optimizer
│   ├── data_drift.py         #   PSI-based data drift detection
│   ├── analytics.py          #   Precomputed analytics loader
│   ├── trend_analytics.py    #   Monthly churn trend computation
│   ├── geographic.py         #   Geographic churn analysis
│   └── model_health.py       #   Model performance monitoring
│
├── models/
│   └── schemas.py            # Pydantic request/response schemas
│
├── utils/
│   └── logging_middleware.py # Request logging middleware
│
├── artifacts/                # Serialized models & precomputed data
│   ├── churn_model.pkl       #   Primary trained pipeline
│   ├── best_model.pkl        #   Best model from comparison
│   ├── segmentation_model.pkl#   K-Means model
│   ├── model_features.pkl    #   Feature list used by the model
│   ├── model_comparison.json #   Comparison results
│   ├── segment_profiles.json #   Segment descriptions
│   └── analytics.json        #   Precomputed analytics
│
├── static/
│   └── index.html            # SaaS frontend (single-page application)
│
└── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/harjotsingh13/churnguard-ai.git
cd churnguard-ai/telecom_saas

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The API will start at **http://localhost:8000**
- 📄 **API Docs (Swagger):** http://localhost:8000/docs
- 🖥️ **Frontend Dashboard:** http://localhost:8000/app

---

## 📡 API Endpoints

### Prediction
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict churn for a single customer (with SHAP explanation + segment) |
| `POST` | `/batch_predict` | Upload CSV for portfolio-wide churn analysis |

### Strategy & Campaigns
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/simulate_strategy` | Simulate discount strategies on a customer portfolio |
| `POST` | `/optimize_campaign` | Budget-constrained retention campaign optimization |

### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/analytics/overview` | High-level KPI summary |
| `GET` | `/analytics/all` | Full analytics payload |
| `GET` | `/analytics/trends` | Monthly churn trend data |
| `GET` | `/analytics/geographic` | Geographic churn breakdown |

### Model & Explainability
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/model_comparison` | 4-model comparison results (AUC, F1, Precision, Recall) |
| `GET` | `/model_health` | Model performance & monitoring status |
| `GET` | `/global_feature_importance` | Global SHAP feature importance |
| `GET` | `/data_drift` | Data drift detection via PSI |
| `GET` | `/segments` | Customer segment profiles |

---

## 🧠 Machine Learning Pipeline

### Training (`train.ipynb`)
1. **Data Preprocessing** — handles missing values, encodes categoricals via `ColumnTransformer`
2. **Model Training** — scikit-learn `Pipeline` with preprocessor + classifier
3. **Artifacts Export** — serialized model, features list, churn rate, analytics

### Model Comparison (`services/model_comparison.py`)
Benchmarks 4 classification models with **5-fold Stratified Cross-Validation**:

| Model | Metrics |
|-------|---------|
| Logistic Regression | AUC, Accuracy, Precision, Recall, F1 |
| Random Forest | AUC, Accuracy, Precision, Recall, F1 |
| XGBoost | AUC, Accuracy, Precision, Recall, F1 |
| LightGBM | AUC, Accuracy, Precision, Recall, F1 |

Generates ROC curves and confusion matrix comparison charts.

### Explainability (`services/shap_explainer.py`)
- **Local:** Per-prediction SHAP values → top drivers & protectors with natural language explanations
- **Global:** Dataset-wide SHAP feature importance ranking
- Aggregates one-hot encoded SHAP values back to original feature names

### Customer Segmentation (`services/segmentation.py`)
- K-Means clustering on behavioral features (tenure, charges, usage)
- 4 business-meaningful segments with tailored retention strategies:
  - **New Price-Sensitive** — short-tenure, low spend
  - **Loyal High-Value** — long-tenure, high spend
  - **At-Risk Mid-Tier** — medium engagement, drifting
  - **Digital Power Users** — high data/streaming usage

---

## 🔍 Data Drift Detection

Uses **Population Stability Index (PSI)** to detect distribution shifts between training data and incoming batches:
- PSI < 0.1 → **Stable** (no significant drift)
- 0.1 ≤ PSI < 0.2 → **Warning** (minor shift)
- PSI ≥ 0.2 → **Drift Detected** (model retraining recommended)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn |
| ML | scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Validation | Pydantic v2 |
| Frontend | Vanilla HTML/CSS/JS |

---

## 📊 Dataset

**IBM Telco Customer Churn** dataset with 7,043 customers and 30+ features including demographics, account info, service subscriptions, and usage metrics.

---

## 📝 License

This project was built as a demonstration of end-to-end ML engineering for academic and interview purposes.
