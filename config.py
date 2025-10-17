import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

class Config:
    """
    Centralized configuration class for the AI Voice Cloning Engine.
    Manages API keys, model names, and other application settings.
    """
    # Application Settings
    APP_NAME: str = "AI Voice Cloning Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # API Keys & Endpoints (Use environment variables for production)
    # Hugging Face Inference API Token
    HUGGINGFACE_API_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN", "hf_YOUR_DEFAULT_TOKEN_HERE")
    
    # Text-to-Speech Models (Hugging Face Inference API)
    # Ensure these are actual Inference API endpoints.
    # We will use facebook/mms-tts for multilingual support
    TTS_MODEL_ENGLISH: str = "facebook/mms-tts-eng" # Example
    TTS_MODEL_GERMAN: str = "facebook/mms-tts-deu" # Example
    TTS_MODEL_URDU: str = "facebook/mms-tts-urd" # Example

    # Speaker Embedding Model (Hugging Face Inference API for feature extraction)
    # We will use a general-purpose audio embedding model.
    # The actual embedding logic will be in core/embedding.py
    EMBEDDING_MODEL_REPO_ID: str = "nvidia/speakerverification_en_titanet_large" # CHANGED THIS LINE

    # Security Settings
    # This should be a strong, randomly generated key in production.
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "super-secret-key-replace-me")
    AUTH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 # Token valid for 24 hours

    # Audio Processing Settings
    SAMPLE_RATE: int = 16000 # Standard sample rate for many ASR/TTS models
    AUDIO_CHUNK_SIZE: int = 1024 # Buffer size for audio processing
    MAX_AUDIO_DURATION_SECONDS: int = 60 # Max 1 minute audio sample for cloning
    MAX_TEXT_LENGTH: int = 500 # Max characters for TTS input

    # Watermarking Settings
    # Frequency range for embedding a watermark (Hz) - needs to be inaudible
    WATERMARK_FREQ_LOW: float = 19000.0
    WATERMARK_FREQ_HIGH: float = 20000.0
    WATERMARK_AMPLITUDE_SCALE: float = 0.005 # Small amplitude to keep it inaudible

    # Paths
    TEMP_AUDIO_DIR: str = "temp_audio"

    class Log:
        """Logging configurations."""
        LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
        FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"