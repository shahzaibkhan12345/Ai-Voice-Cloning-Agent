# app/services/audio_preprocessor.py
import io
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from app.utils.exceptions import AudioProcessingError

def preprocess_audio(audio_bytes: bytes) -> bytes:
    """
    Preprocesses audio bytes to ensure quality for the API.
    - Converts to WAV format.
    - Converts to mono.
    - Sets sample rate to 44100 Hz (standard for ElevenLabs).
    - Normalizes volume.
    
    Args:
        audio_bytes: The raw audio data from the uploaded file.

    Returns:
        The processed audio data in WAV format as bytes.
        
    Raises:
        AudioProcessingError: If the audio cannot be processed.
    """
    try:
        # Load audio from the in-memory bytes
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

        # Convert to mono to ensure a single channel
        audio = audio.set_channels(1)

        # Set sample rate to 44100 Hz for best compatibility
        audio = audio.set_frame_rate(44100)

        # Normalize the audio to a standard volume level (e.g., -20 dBFS)
        # This prevents the output from being too loud or too quiet.
        audio = audio.normalize(headroom=20.0)

        # Export the processed audio back to bytes in WAV format
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        
        return wav_buffer.getvalue()

    except CouldntDecodeError:
        raise AudioProcessingError("Could not decode audio file. Please ensure it's a valid audio format (WAV, MP3, etc.).")
    except Exception as e:
        raise AudioProcessingError(f"An unexpected error occurred during audio preprocessing: {e}")
