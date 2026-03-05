"""
Logging middleware for FastAPI.
Logs every request with method, path, status code, and duration.
Extra detail for prediction and campaign optimization calls.
"""

import logging
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("churnguard.middleware")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        method = request.method
        path = request.url.path

        # Process request
        response = await call_next(request)

        duration_ms = (time.perf_counter() - start) * 1000
        status = response.status_code

        # Standard log line
        logger.info(
            f"{method} {path} → {status} ({duration_ms:.0f}ms)"
        )

        # Extra logging for key endpoints
        if path == "/predict" and method == "POST":
            logger.info("  📊 Individual churn prediction executed")
        elif path == "/optimize_campaign" and method == "POST":
            logger.info("  🎯 Campaign optimization executed")
        elif path == "/batch_predict" and method == "POST":
            logger.info("  📁 Batch portfolio prediction executed")

        return response
