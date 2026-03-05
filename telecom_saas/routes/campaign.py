"""
Campaign Optimization route — POST /optimize_campaign
"""

import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Query

import pandas as pd

from services.campaign_optimizer import optimize_campaign

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Campaign"])


@router.post("/optimize_campaign")
async def run_campaign(
    file: UploadFile = File(...),
    budget: float = Query(..., description="Total retention budget in dollars"),
    discount_per_customer: float = Query(10, description="Discount per customer in dollars"),
):
    try:
        df = pd.read_csv(file.file)
        result = optimize_campaign(df, budget, discount_per_customer)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
