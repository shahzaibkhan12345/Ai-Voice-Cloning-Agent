import os
import asyncio
import numpy as np
import requests
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from config import Config
from api.utils import get_logger

logger = get_logger(__name__)

HUGGINGFACE_API_URL = f"https://api-inference.huggingface.co/models/{Config.EMBEDDING_MODEL_REPO_ID}"
HEADERS = {"Authorization": f"Bearer {Config.HUGGINGFACE_API_TOKEN}"}

async def _call_huggingface_embedding_api(audio_bytes: bytes) -> Optional[Any]: # Changed return type to Any for flexibility
    """
    Internal asynchronous function to call the Hugging Face Inference API for embeddings.
    Runs the blocking requests.post in a separate thread.
    """
    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(
            None, # Use the default ThreadPoolExecutor
            lambda: requests.post(HUGGINGFACE_API_URL, headers=HEADERS, data=audio_bytes)
        )
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        json_response = response.json()
        logger.debug(f"Hugging Face embedding API raw response: {json_response}")
        return json_response

    except requests.exceptions.RequestException as e:
        logger.error(f"Hugging Face API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from Hugging Face API: {e}")
        logger.debug(f"Raw response content: {response.text if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while calling Hugging Face API: {e}")
        return None


async def generate_embedding(audio_path: Path) -> Optional[np.ndarray]:
    """
    Generates a speaker embedding (voice vector) from a preprocessed audio file
    using the Hugging Face Inference API.

    Args:
        audio_path: Path to the preprocessed mono WAV audio file (e.g., 16kHz).

    Returns:
        A numpy array representing the speaker embedding, or None if generation fails.
    """
    logger.info(f"Generating speaker embedding for {audio_path} using {Config.EMBEDDING_MODEL_REPO_ID}...")

    if not audio_path.exists():
        logger.error(f"Audio file for embedding not found: {audio_path}")
        return None

    try:
        # Read the audio file bytes
        audio_bytes = audio_path.read_bytes()

        # Call the Hugging Face Inference API
        api_response = await _call_huggingface_embedding_api(audio_bytes)

        if api_response is None:
            logger.error("Failed to get a response from the Hugging Face embedding API.")
            return None

        # --- IMPORTANT: Parsing the response for nvidia/speakerverification_en_titanet_large ---
        # The exact structure can vary. We expect a list of floats (the embedding vector)
        # It might be directly the response, or nested under a key.
        embedding = None
        
        if isinstance(api_response, list) and all(isinstance(x, (float, int)) for x in api_response):
            # If the response is directly a list of floats (the embedding vector)
            embedding = np.array(api_response, dtype=np.float32)
        elif isinstance(api_response, dict) and "embedding" in api_response and \
             isinstance(api_response["embedding"], list) and \
             all(isinstance(x, (float, int)) for x in api_response["embedding"]):
            # If the response is a dict with an "embedding" key containing the list of floats
            embedding = np.array(api_response["embedding"], dtype=np.float32)
        elif isinstance(api_response, list) and len(api_response) > 0 and \
             isinstance(api_response[0], list) and \
             all(isinstance(x, (float, int)) for x in api_response[0]):
            # If it's a list of lists, and the first inner list is the embedding (e.g., [[embedding_vector]])
            # This can happen if the API returns a batch of 1. Take the first.
            embedding = np.array(api_response[0], dtype=np.float32)
        else:
            logger.error(f"Unexpected or unparseable API response format for embedding from {Config.EMBEDDING_MODEL_REPO_ID}: {api_response}")
            return None
        
        logger.info(f"Speaker embedding generated successfully. Shape: {embedding.shape}")
        return embedding

    except Exception as e:
        logger.exception(f"An error occurred during speaker embedding generation for {audio_path}: {e}")
        return None