"""
Analytics routes — GET /analytics/overview, /analytics/all, /analytics/trends, /analytics/geographic
"""

import logging
from fastapi import APIRouter, HTTPException

from services.analytics import load_analytics
from services.trend_analytics import compute_trends
from services.geographic import compute_geographic

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/overview")
def analytics_overview():
    analytics = load_analytics()
    return analytics["overview"]


@router.get("/all")
def analytics_all():
    return load_analytics()


@router.get("/trends")
def analytics_trends():
    try:
        return compute_trends()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/geographic")
def analytics_geographic():
    try:
        return compute_geographic()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
