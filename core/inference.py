import os
import asyncio
import requests
import numpy as np
import soundfile as sf
import io
from pathlib import Path
from typing import Optional, Tuple, Dict
import librosa # For calculating audio duration reliably

from config import Config
from api.utils import get_logger

logger = get_logger(__name__)

# Base URL for Hugging Face Inference API
HUGGINGFACE_INFERENCE_API_BASE = "https://api-inference.huggingface.co/models/"
HEADERS = {"Authorization": f"Bearer {Config.HUGGINGFACE_API_TOKEN}"}

# Mapping of language to Hugging Face TTS model ID
TTS_MODEL_MAP: Dict[str, str] = {
    "english": Config.TTS_MODEL_ENGLISH,  # e.g., facebook/mms-tts-eng
    "german": Config.TTS_MODEL_GERMAN,    # e.g., facebook/mms-tts-deu
    "urdu": Config.TTS_MODEL_URDU,        # e.g., facebook/mms-tts-urd
}

async def _call_huggingface_tts_api(text: str, model_id: str) -> Optional[bytes]:
    """
    Internal asynchronous function to call the Hugging Face Inference API for TTS.
    Runs the blocking requests.post in a separate thread.

    Args:
        text: The text to synthesize.
        model_id: The Hugging Face model ID to use for TTS.

    Returns:
        Raw audio bytes if successful, None otherwise.
    """
    api_url = f"{HUGGINGFACE_INFERENCE_API_BASE}{model_id}"
    payload = {"inputs": text}
    
    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(
            None, # Use the default ThreadPoolExecutor
            lambda: requests.post(api_url, headers=HEADERS, json=payload)
        )
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        # Hugging Face TTS Inference API for `text-to-speech` task returns raw audio bytes
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Hugging Face TTS API request failed for model {model_id}: {e}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while calling Hugging Face TTS API for {model_id}: {e}")
        return None

async def synthesize(
    text: str,
    language: str,
    speaker_embedding: Optional[np.ndarray], # This is kept for future-proofing, see explanation
    output_audio_path: Path
) -> Tuple[Optional[Path], float]:
    """
    Synthesizes text into speech in the specified language.
    
    NOTE: Currently, the public Hugging Face Inference API for models like `facebook/mms-tts`
    does not support direct injection of external speaker embeddings for arbitrary voice cloning.
    The `speaker_embedding` parameter is included for architectural completeness,
    but it will be unused in this implementation. The output will be in the
    default voice provided by the selected TTS model for the given language.

    Args:
        text: The text to be synthesized.
        language: The target language for synthesis ("english", "german", "urdu").
        speaker_embedding: The numpy array representing the speaker's voice (currently unused).
        output_audio_path: The path where the synthesized audio WAV file will be saved.

    Returns:
        A tuple containing:
        - Path to the synthesized audio file if successful, None otherwise.
        - Duration of the synthesized audio in seconds, 0.0 if synthesis fails.
    """
    logger.info(f"Starting speech synthesis for language '{language}' (text length: {len(text)}).")

    model_id = TTS_MODEL_MAP.get(language)
    if not model_id:
        logger.error(f"Unsupported language for TTS: {language}. No model mapped.")
        return None, 0.0

    logger.debug(f"Using Hugging Face model: {model_id} for synthesis.")

    try:
        audio_bytes = await _call_huggingface_tts_api(text, model_id)

        if audio_bytes is None or not audio_bytes:
            logger.error(f"Failed to get audio bytes from TTS API for model {model_id}.")
            return None, 0.0

        # Hugging Face TTS inference API typically returns raw audio that is WAV formatted.
        # We need to save it and then load with librosa to get duration reliably.
        with open(output_audio_path, "wb") as f:
            f.write(audio_bytes)
        logger.debug(f"Raw synthesized audio saved to {output_audio_path}")

        # Calculate duration using librosa.load with sf (soundfile) backend
        # Use io.BytesIO to avoid saving to disk and reloading, more efficient
        # Re-read from path for simplicity if `soundfile` with BytesIO gets complex
        audio_data, sr = librosa.load(str(output_audio_path), sr=None, mono=True)
        duration_seconds = librosa.get_duration(y=audio_data, sr=sr)
        
        logger.info(f"Speech synthesis complete. Audio saved to {output_audio_path}, duration: {duration_seconds:.2f}s")
        return output_audio_path, duration_seconds

    except Exception as e:
        logger.exception(f"An error occurred during speech synthesis for text '{text[:50]}...' and language '{language}': {e}")
        return None, 0.0