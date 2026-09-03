"""
main.py - FastAPI application entrypoint for Onnano AI/CV Object Understanding & Comparison Engine.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import analyze_router, compare_router

app = FastAPI(
    title="Onnano - AI/CV Object Understanding & Comparison Engine",
    description="Computer-vision and AI pipeline for object detection, posture normalization, physical property analysis, geometric division, and multi-object comparison.",
    version="2.0.0",
)

# CORS Middleware to support local dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "onnano-cv-ai-engine",
        "version": "2.0.0",
        "pipeline": ["segmentation", "orientation_normalization", "property_extraction", "slicing", "comparison"],
    }


app.include_router(analyze_router)
app.include_router(compare_router)
