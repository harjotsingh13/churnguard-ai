"""
Segments route — GET /segments
"""

import logging
from fastapi import APIRouter, HTTPException

from services.segmentation import load_segment_profiles

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Segmentation"])


@router.get("/segments")
def get_segments():
    result = load_segment_profiles()
    if result is None:
        raise HTTPException(status_code=404, detail="Segmentation not available. Run segmentation.py first.")
    return result
