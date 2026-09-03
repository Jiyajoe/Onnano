"""
main.py - FastAPI entrypoint for the "Onnaano?" AI Sibling Fairness Judge.

Run with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import analyze, verify

app = FastAPI(
    title="Onnaano? - AI Sibling Fairness Judge",
    description="Computer-vision powered referee for fairly dividing shared objects between siblings.",
    version="1.0.0",
)

# Local dev: Vite's default port plus a couple of common alternates.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never leak raw stack traces to the client (spec section 24).
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "Something went wrong while processing the image. Please try again."},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "onnaano-fairness-judge"}


app.include_router(analyze.router)
app.include_router(verify.router)
