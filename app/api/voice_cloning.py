# app/api/voice_cloning.py
import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import StreamingResponse

from app.services.audio_preprocessor import preprocess_audio
from app.services.voice_synthesizer import VoiceSynthesizer
from app.services.watermarking import add_watermark
from app.core.config import settings
from app.utils.exceptions import (
    AudioProcessingError,
    SynthesisError,
    WatermarkingError
)

# Create a FastAPI router
router = APIRouter(
    prefix=settings.API_V1_STR,
    tags=["Voice Cloning"]
)

# Instantiate the synthesizer service once to be reused across requests
synthesizer = VoiceSynthesizer()

@router.post("/clone", status_code=status.HTTP_200_OK)
async def clone_voice_endpoint(
    text: str = Form(..., min_length=1, max_length=5000, example="Hello, this is a test of the voice cloning engine."),
    language: str = Form(default="en", example="en"),
    voice_sample: UploadFile = File(..., description="A clean voice sample (WAV/MP3, ~10-30 seconds).")
):
    """
    Accepts a voice sample and text, and returns audio of the text spoken in the cloned voice.
    
    - **text**: The text you want the cloned voice to speak.
    - **language**: The language of the text (ISO 639-1 code).
    - **voice_sample**: An audio file containing the voice to be cloned.
    """
    # 1. Validate the language
    if language not in settings.SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported language '{language}'. Supported languages are: {settings.SUPPORTED_LANGUAGES}"
        )

    # 2. Validate the uploaded file type
    if not voice_sample.content_type or not voice_sample.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload an audio file."
        )

    try:
        # 3. Read and preprocess the audio sample
        voice_sample_bytes = await voice_sample.read()
        processed_audio_bytes = preprocess_audio(voice_sample_bytes)

        # 4. Synthesize new speech using the ElevenLabs API
        synthesized_audio_bytes = synthesizer.clone_voice(
            text=text,
            voice_sample_bytes=processed_audio_bytes
        )

        # 5. Add the security watermark to the output
        watermarked_audio_bytes = add_watermark(synthesized_audio_bytes)

        # 6. Return the final audio file as a streaming response
        # This is more memory-efficient for larger files
        return StreamingResponse(
            io.BytesIO(watermarked_audio_bytes),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=cloned_voice.mp3"}
        )

    # 7. Handle our custom, specific exceptions
    except AudioProcessingError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio processing error: {e}"
        )
    except SynthesisError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Voice synthesis service error: {e}"
        )
    except WatermarkingError as e:
        # This should not happen as we return original audio on failure, but just in case.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Watermarking error: {e}"
        )
    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}"
        )
