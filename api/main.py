"""
FastAPI main app for /api/clone endpoint.
"""

from fastapi import FastAPI

app = FastAPI(title="AI Voice Cloning Engine")


@app.get("/")
def root():
    return {"message": "AI Voice Cloning Engine API is running."}
