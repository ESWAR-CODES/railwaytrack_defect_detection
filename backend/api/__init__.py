"""Railway Defect Inspection FastAPI Package.

Mohan Babu University - Dept. of CSE (AI & ML) - Team B6-B3.
Phase 4: FastAPI Inference & Inspection Backend.
"""

from backend.api.main import app
from backend.api.routes import router

__all__ = ["app", "router"]
