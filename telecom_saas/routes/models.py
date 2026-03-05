"""
Model routes — GET /model_comparison, /model_health
"""

import logging
from fastapi import APIRouter, HTTPException

from services.model_comparison import load_comparison_results
from services.model_health import get_model_health

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Model"])


@router.get("/model_comparison")
def model_comparison():
    result = load_comparison_results()
    if result is None:
        raise HTTPException(status_code=404, detail="Model comparison not available. Run model_comparison.py first.")
    return result


@router.get("/model_health")
def model_health():
    result = get_model_health()
    if result is None:
        raise HTTPException(status_code=404, detail="Model health data not available.")
    return result
