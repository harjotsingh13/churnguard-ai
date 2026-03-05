"""
ChurnGuard AI — Customer Intelligence & Revenue Optimization API
Main application entry point. Mounts routers and middleware.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from utils.logging_middleware import RequestLoggingMiddleware
from routes import prediction, portfolio, strategy, analytics, models, segments, campaign, explainability

# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("churnguard")

# --------------------------------------------------
# FastAPI Initialization
# --------------------------------------------------

app = FastAPI(
    title="ChurnGuard AI — Customer Intelligence & Revenue Optimization API",
    description="Churn Prediction, LTV Revenue Risk, Portfolio Analytics, Strategy Simulation, Campaign Optimization & Model Monitoring",
    version="3.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Logging Middleware
app.add_middleware(RequestLoggingMiddleware)

# --------------------------------------------------
# Mount Routers
# --------------------------------------------------

app.include_router(prediction.router)
app.include_router(portfolio.router)
app.include_router(strategy.router)
app.include_router(analytics.router)
app.include_router(models.router)
app.include_router(segments.router)
app.include_router(campaign.router)
app.include_router(explainability.router)

# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/", tags=["Health"])
def health_check():
    return {"status": "API is running successfully", "version": "3.0.0"}

# --------------------------------------------------
# Serve Frontend
# --------------------------------------------------

@app.get("/app", tags=["Frontend"])
def serve_frontend():
    return FileResponse("static/index.html")

# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
