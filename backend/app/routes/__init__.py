"""
Routes package initialization
"""
from .analyze import router as analyze_router
from .compare import router as compare_router

__all__ = ["analyze_router", "compare_router"]
