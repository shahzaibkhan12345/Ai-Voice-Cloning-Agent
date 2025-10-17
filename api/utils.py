import logging
from typing import Any
from config import Config

# --- Logging Setup ---
def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger instance with a standardized format.

    Args:
        name: The name of the logger (typically __name__ of the module).

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers: # Prevent adding multiple handlers if called multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt=Config.Log.FORMAT,
            datefmt=Config.Log.DATE_FORMAT
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(Config.Log.LEVEL)
    return logger

logger = get_logger(__name__)

# --- Helper Functions / Validators (Example) ---
def validate_audio_file(file_content: bytes, max_size_mb: int = 5) -> None:
    """
    Validates the size of an uploaded audio file.

    Args:
        file_content: The byte content of the uploaded file.
        max_size_mb: The maximum allowed file size in megabytes.

    Raises:
        ValueError: If the file size exceeds the maximum allowed.
    """
    file_size_bytes = len(file_content)
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > max_size_mb:
        logger.error(f"File size {file_size_mb:.2f}MB exceeds maximum allowed {max_size_mb}MB.")
        raise ValueError(f"Audio file too large. Max allowed is {max_size_mb}MB.")
    logger.debug(f"Audio file size: {file_size_mb:.2f}MB (within {max_size_mb}MB limit).")

def validate_text_input(text: str, max_length: int = Config.MAX_TEXT_LENGTH) -> None:
    """
    Validates the length of the input text for TTS.

    Args:
        text: The input text string.
        max_length: The maximum allowed length of the text.

    Raises:
        ValueError: If the text length exceeds the maximum allowed.
    """
    if not text or not text.strip():
        logger.error("Input text cannot be empty.")
        raise ValueError("Input text cannot be empty.")
    if len(text) > max_length:
        logger.error(f"Input text length {len(text)} exceeds maximum allowed {max_length}.")
        raise ValueError(f"Input text too long. Max allowed is {max_length} characters.")
    logger.debug(f"Input text length: {len(text)} characters (within {max_length} limit).")

# You can add more utility functions here as needed,
# e.g., for data serialization/deserialization, API response formatting, etc.