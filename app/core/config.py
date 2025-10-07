# app/core/config.py
import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """
    Loads and validates application settings from environment variables.
    """
    # Application Metadata
    APP_NAME: str = "Voice Cloning API"
    APP_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # ElevenLabs Configuration
    ELEVENLABS_API_KEY: str
    
    # Model to use for instant voice cloning
    ELEVENLABS_MODEL_ID: str = "eleven_multilingual_v2"

    # Supported Languages (ISO 639-1 codes)
    SUPPORTED_LANGUAGES: list = ["en", "es", "de", "fr", "hi", "ur", "pt"]

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached instance of the Settings.
    This ensures the .env file is read only once.
    """
    return Settings()

# Create a single settings instance to be imported by other modules
settings = get_settings()