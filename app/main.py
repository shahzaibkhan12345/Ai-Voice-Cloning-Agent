# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.voice_cloning import router as voice_cloning_router
from app.core.config import settings

# Create the FastAPI application instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="A production-ready AI Voice Cloning Engine API built with FastAPI and ElevenLabs.",
    docs_url="/docs",  # URL for Swagger UI
    redoc_url="/redoc" # URL for ReDoc
)

# Configure CORS (Cross-Origin Resource Sharing) middleware
# This allows our API to be called from web browsers on different domains.
# For development, we allow all origins. In production, you should restrict this!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

# Include the API router from our voice cloning module
app.include_router(voice_cloning_router)

# Define a simple root endpoint to check if the API is running
@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint to welcome users and provide basic info.
    """
    return {
        "message": f"Welcome to the {settings.APP_NAME} API.",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

# To run this application:
# 1. Make sure your virtual environment is active.
# 2. Run the command: uvicorn app.main:app --reload
#    - `uvicorn` is the ASGI server.
#    - `app.main:app` tells uvicorn where to find the FastAPI app instance (in the `main.py` file, inside the `app` module, named `app`).
#    - `--reload` makes the server restart automatically when you change the code.