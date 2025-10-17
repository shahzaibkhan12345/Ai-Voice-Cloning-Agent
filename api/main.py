import os
import shutil
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import Config
from api.utils import get_logger, validate_audio_file, validate_text_input
from api.auth import verify_token
# Import core modules (will be implemented next)
from core.preprocess import preprocess_audio
from core.embedding import generate_embedding
from core.inference import synthesize
from core.watermark import embed_watermark

# Initialize logger
logger = get_logger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title=Config.APP_NAME,
    version=Config.APP_VERSION,
    debug=Config.DEBUG,
    description="A production-grade AI Voice Cloning Engine",
)

# --- Ensure temporary audio directory exists ---
TEMP_AUDIO_PATH = Path(Config.TEMP_AUDIO_DIR)
TEMP_AUDIO_PATH.mkdir(parents=True, exist_ok=True)
logger.info(f"Temporary audio directory ensured: {TEMP_AUDIO_PATH}")

# --- Pydantic Models for Request/Response ---
class CloneRequest(BaseModel):
    """
    Schema for the /api/clone endpoint request body (when not using form-data for audio).
    """
    text: str = Field(..., max_length=Config.MAX_TEXT_LENGTH, 
                      example="Hello, this is a cloned voice speaking in English.")
    language: str = Field(..., pattern="^(english|german|urdu)$", 
                          example="english", description="Target language for synthesis (english, german, urdu)")

class CloneResponse(BaseModel):
    """
    Schema for the /api/clone endpoint response.
    Note: The actual audio will be streamed, this is for metadata/error response.
    """
    message: str = Field(..., example="Voice cloning successful.")
    audio_format: str = Field("wav", example="wav")
    duration_seconds: float = Field(..., example=5.23)
    # In a real app, you might return a URL to the audio file if not streaming directly

# --- Health Check Endpoint ---
@app.get("/health", response_class=JSONResponse, summary="Health Check")
async def health_check():
    """
    Checks the health of the API.
    """
    logger.info("Health check endpoint accessed.")
    return {"status": "ok", "version": Config.APP_VERSION, "name": Config.APP_NAME}

# --- Main Voice Cloning Endpoint ---
@app.post(
    "/api/clone",
    summary="Clone Voice and Synthesize Text",
    description="Accepts an audio sample and text, then returns synthesized speech in the cloned voice.",
    response_description="Synthesized audio in WAV format.",
    status_code=status.HTTP_200_OK,
    # Here, we indicate that the response will be audio/wav, not a JSON response model directly.
    # We could define a custom MediaResponse class for more explicit OpenAPI docs.
)
async def clone_voice_and_synthesize(
    # File upload via Form Data (audio_sample)
    audio_sample: Annotated[UploadFile, File(description="1-minute WAV/MP3 audio sample for voice cloning")],
    # Text input via Form Data
    text: Annotated[str, Form(max_length=Config.MAX_TEXT_LENGTH, 
                              example="Guten Tag, wie geht es Ihnen heute?",
                              description="Text to convert into speech.")],
    # Language input via Form Data
    language: Annotated[str, Form(pattern="^(english|german|urdu)$", 
                                  example="german",
                                  description="Target language for synthesis (english, german, urdu).")],
    # Authentication Dependency
    authenticated_user: Annotated[dict, Depends(verify_token)] # Requires a valid JWT
) -> StreamingResponse: # Explicitly hint that we return a StreamingResponse
    """
    Endpoint to clone a voice from an audio sample and synthesize text into that voice.

    - **audio_sample**: An audio file (WAV or MP3, max 1 minute) of the voice to clone.
    - **text**: The text to be converted into speech in the cloned voice.
    - **language**: The target language for synthesis (English, German, or Urdu).
    - **Requires authentication** with a valid Bearer token.
    """
    request_id = authenticated_user.get("sub", "unknown_user") # Get user ID from token
    logger.info(f"[{request_id}] /api/clone endpoint accessed. Language: {language}, Text length: {len(text)}")

    # 1. Input Validation
    try:
        audio_content = await audio_sample.read()
        validate_audio_file(audio_content, max_size_mb=5) # 5MB max
        validate_text_input(text)

        if not language in ["english", "german", "urdu"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported language. Supported languages are 'english', 'german', 'urdu'."
            )
        logger.debug(f"[{request_id}] Input validation passed.")

    except ValueError as e:
        logger.error(f"[{request_id}] Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"[{request_id}] An unexpected error occurred during input validation.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")

    # Generate a unique filename for temporary audio files
     # Generate a unique filename for temporary audio files
    audio_filename_base = f"{request_id}_{os.urandom(8).hex()}"
    
    # Initialize all path variables here, so they are always defined
    # Even if the file is not created, the Path object itself exists.
    input_audio_path = TEMP_AUDIO_PATH / f"{audio_filename_base}_input.wav"
    processed_audio_path = TEMP_AUDIO_PATH / f"{audio_filename_base}_processed.wav"
    output_audio_path = TEMP_AUDIO_PATH / f"{audio_filename_base}_output.wav"
    watermarked_audio_path = TEMP_AUDIO_PATH / f"{audio_filename_base}_watermarked.wav" # Defined here

    try:
        # Save the uploaded audio to a temporary file for processing
        with open(input_audio_path, "wb") as f:
            f.write(audio_content)
        logger.debug(f"[{request_id}] Input audio saved to {input_audio_path}")

        # 2. Preprocess Audio Sample
        logger.info(f"[{request_id}] Starting audio preprocessing...")
        await preprocess_audio(input_audio_path, processed_audio_path)
        logger.info(f"[{request_id}] Audio preprocessing complete: {processed_audio_path}")

        # 3. Generate Speaker Embedding
        logger.info(f"[{request_id}] Generating speaker embedding...")
        speaker_embedding = await generate_embedding(processed_audio_path)
        if speaker_embedding is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate speaker embedding. Please try again."
            )
        logger.info(f"[{request_id}] Speaker embedding generated.")

        # 4. Synthesize Text to Speech
        logger.info(f"[{request_id}] Synthesizing speech...")
        synthesized_audio_path, duration = await synthesize(text, language, speaker_embedding, output_audio_path)
        if synthesized_audio_path is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to synthesize speech. Please try again or check the input text/language."
            )
        logger.info(f"[{request_id}] Speech synthesis complete. Duration: {duration:.2f}s")

        # 5. Embed Watermark (Optional but good for tracking)
        logger.info(f"[{request_id}] Embedding watermark...")
        watermarked_audio_path = TEMP_AUDIO_PATH / f"{audio_filename_base}_watermarked.wav"
        await embed_watermark(synthesized_audio_path, watermarked_audio_path, request_id)
        logger.info(f"[{request_id}] Watermark embedded.")
        
        # Prepare for streaming the audio back
        def iterfile():
            with open(watermarked_audio_path, mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(
            iterfile(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=cloned_voice_{request_id}.wav",
                "X-Audio-Duration": str(duration),
                "X-Generated-By": Config.APP_NAME
            }
        )

    except HTTPException:
        # Re-raise HTTPException directly as it's already structured for FastAPI
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] An unexpected error occurred during voice cloning: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred: {e}"
        )
    finally:
        # Clean up temporary files
        # Check if the Path object exists on disk before attempting to remove
        for f_path in [input_audio_path, processed_audio_path, output_audio_path, watermarked_audio_path]:
            if f_path.exists():
                os.remove(f_path)
                logger.debug(f"[{request_id}] Cleaned up temporary file: {f_path}")