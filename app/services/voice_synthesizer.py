# app/services/voice_synthesizer.py
import io
from elevenlabs import VoiceSettings, ElevenLabs  # <-- CORRECTED IMPORT
from app.core.config import settings
from app.utils.exceptions import SynthesisError

class VoiceSynthesizer:
    """
    A service class to handle voice synthesis using the ElevenLabs API.
    """
    def __init__(self):
        """Initializes the synthesizer with the API key from settings."""
        if not settings.ELEVENLABS_API_KEY:
            raise SynthesisError("ELEVENLABS_API_KEY is not configured.")
        self.client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

    def clone_voice(self, text: str, voice_sample_bytes: bytes) -> bytes:
        """
        Clones a voice from a sample and synthesizes new speech.

        Args:
            text: The text to synthesize.
            voice_sample_bytes: The processed voice sample audio data in bytes.

        Returns:
            The synthesized audio data in bytes (MP3 format).
            
        Raises:
            SynthesisError: If the API call fails.
        """
        try:
            # Define voice settings for stability and clarity
            voice_settings = VoiceSettings(
                stability=0.71,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            )
            
            # The ElevenLabs API can directly use a voice sample for instant cloning
            response = self.client.text_to_speech.convert(
                voice_id="custom", # Use a placeholder voice_id
                text=text,
                model_id=settings.ELEVENLABS_MODEL_ID,
                voice_settings=voice_settings,
                # The magic happens here: we provide the audio sample directly
                voice_sample=io.BytesIO(voice_sample_bytes)
            )

            # The API response is an iterator of audio chunks. We join them.
            audio_bytes = b"".join(chunk for chunk in response)
            return audio_bytes

        except Exception as e:
            # Catch any exception from the ElevenLabs client and re-raise it as our custom error
            raise SynthesisError(f"ElevenLabs API call failed: {e}")
