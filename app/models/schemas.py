# app/models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class VoiceCloneRequest(BaseModel):
    """
    Schema for the voice cloning request.
    This model validates the data sent to our API endpoint.
    """
    text: str = Field(..., min_length=1, max_length=5000, example="Hello, this is a test of the voice cloning engine.")
    language: str = Field(default="en", example="en")

class VoiceCloneResponse(BaseModel):
    """
    Schema for the successful voice cloning response.
    """
    message: str = Field(default="Voice synthesized successfully.")
    audio_url: Optional[str] = Field(default=None, example="http://localhost:8000/api/v1/audio/output.mp3")

class ErrorResponse(BaseModel):
    """
    Schema for error responses.
    """
    error: str = Field(..., example="SynthesisError")
    message: str = Field(..., example="The ElevenLabs API call failed due to an invalid API key.")