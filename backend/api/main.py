"""FastAPI Application Entrypoint for Railway Defect Monitoring System.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 4: FastAPI Inference & Inspection Backend.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import get_inference_service, router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("railway_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for application initialization and teardown."""
    logger.info("Initializing Railway Defect Detection API service...")
    try:
        service = get_inference_service()
        model_status = "Loaded" if service.model is not None else "Mock/Fallback Mode"
        logger.info(
            f"InferenceService initialized successfully. Active model status: {model_status}. "
            f"Classes: {list(service.config.classes.values())}"
        )
    except Exception as e:
        logger.warning(f"Non-fatal warning during InferenceService pre-load: {e}")

    yield

    logger.info("Railway Defect Detection API shutting down gracefully.")


app = FastAPI(
    title="AI-Based Multi-Defect Railway Monitoring API",
    description=(
        "Backend inference, severity assessment, and telemetry API for railway drone inspection. "
        "Mohan Babu University, Dept. of CSE (AI & ML), Team B6-B3."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure Cross-Origin Resource Sharing (CORS) for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API v1 router
app.include_router(router)


@app.get("/", tags=["Root"])
def root() -> Dict[str, str]:
    """Root endpoint providing service metadata and API documentation references."""
    return {
        "service": "AI-Based Multi-Defect Railway Monitoring API",
        "institution": "Mohan Babu University - Dept. of CSE (AI & ML)",
        "team": "Team B6-B3",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "health_url": "/api/v1/health",
        "metrics_url": "/api/v1/metrics",
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Standardized error handler for client-side HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global unhandled exception safety barrier returning structured JSON."""
    logger.exception(f"Unhandled server error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred while processing the inspection request.",
            "error_type": type(exc).__name__,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
