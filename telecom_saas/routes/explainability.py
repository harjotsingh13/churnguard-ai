"""
Explainability routes — GET /global_feature_importance, /data_drift
"""

import logging
from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Explainability"])


@router.get("/global_feature_importance")
def global_feature_importance():
    try:
        from services.shap_explainer import compute_global_feature_importance
        return {"features": compute_global_feature_importance()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data_drift")
def data_drift():
    try:
        from services.data_drift import check_drift
        return check_drift()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
