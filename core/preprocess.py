import os
import io
import asyncio
import numpy as np
import soundfile as sf
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa # For more advanced signal processing if needed

from config import Config
from api.utils import get_logger

logger = get_logger(__name__)

async def preprocess_audio(input_audio_path: Path, output_audio_path: Path) -> None:
    """
    Performs audio preprocessing steps:
    1. Loads audio (handles various formats via pydub).
    2. Resamples to a consistent sample rate (Config.SAMPLE_RATE).
    3. Normalizes audio to a target loudness (e.g., -20 dBFS).
    4. Trims silence from the beginning and end.
    5. Ensures maximum duration.
    6. Saves the processed audio as a WAV file.

    Args:
        input_audio_path: Path to the raw input audio file (WAV/MP3).
        output_audio_path: Path where the processed WAV file will be saved.

    Raises:
        ValueError: If audio processing fails or input audio is invalid.
        FileNotFoundError: If the input_audio_path does not exist.
        Exception: For any other unexpected errors during processing.
    """
    logger.info(f"Starting audio preprocessing for {input_audio_path}...")

    if not input_audio_path.exists():
        logger.error(f"Input audio file not found: {input_audio_path}")
        raise FileNotFoundError(f"Input audio file not found: {input_audio_path}")

    try:
        # Pydub can load various formats
        # Run pydub operations in a separate thread to not block the event loop
        loop = asyncio.get_running_loop()
        audio_segment = await loop.run_in_executor(
            None, AudioSegment.from_file, str(input_audio_path)
        )

        # 1. Resample (if necessary) and convert to mono (if stereo)
        # Pydub's set_frame_rate and set_channels are blocking, good for executor
        if audio_segment.frame_rate != Config.SAMPLE_RATE:
            audio_segment = await loop.run_in_executor(
                None, audio_segment.set_frame_rate, Config.SAMPLE_RATE
            )
            logger.debug(f"Resampled audio to {Config.SAMPLE_RATE} Hz.")
        
        if audio_segment.channels != 1:
            audio_segment = await loop.run_in_executor(
                None, audio_segment.set_channels, 1
            )
            logger.debug("Converted audio to mono.")

        # 2. Normalize loudness
        # Target loudness around -20 dBFS (decibels relative to full scale)
        # Pydub's normalize is blocking
        normalized_audio = await loop.run_in_executor(None, audio_segment.normalize)
        logger.debug("Normalized audio loudness.")

        # 3. Trim leading/trailing silence (using librosa or pydub can be tricky)
        # For simplicity and robust async, we'll convert to numpy array first
        # and use librosa for trimming silence.
        # This requires `soundfile` to convert pydub to numpy safely.

        # Convert pydub AudioSegment to raw PCM data then to numpy array
        raw_audio_data = np.array(normalized_audio.get_array_of_samples())
        if normalized_audio.sample_width == 2: # 16-bit audio
            audio_array = raw_audio_data.astype(np.int16)
        elif normalized_audio.sample_width == 4: # 32-bit audio
            audio_array = raw_audio_data.astype(np.int32)
        else:
            logger.warning(f"Unsupported sample width: {normalized_audio.sample_width}. Skipping direct numpy conversion for silence trimming.")
            # Fallback: Export to a BytesIO object and load with soundfile
            # This is less efficient but handles various pydub sample_widths to numpy
            with io.BytesIO() as audio_bytes:
                await loop.run_in_executor(None, normalized_audio.export, audio_bytes, format="wav")
                audio_bytes.seek(0)
                audio_array, sr = sf.read(audio_bytes)
                if sr != Config.SAMPLE_RATE: # Should already be correct from pydub.set_frame_rate
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=Config.SAMPLE_RATE)
        
        # Ensure float type for librosa
        if audio_array.dtype != np.float32:
            audio_array = librosa.util.normalize(audio_array.astype(float)) # Convert to float and normalize to [-1, 1]

        # Use librosa for efficient silence trimming
        # top_db=60 is a common threshold for silence
        # frame_length and hop_length affect sensitivity
        trimmed_audio_array, _ = await loop.run_in_executor(
            None, 
            lambda: librosa.effects.trim(audio_array, top_db=60, frame_length=2048, hop_length=512)
        )
        logger.debug("Trimmed leading/trailing silence using librosa.")

        # Reconstruct AudioSegment from trimmed numpy array
        # Librosa's output is float, pydub expects int for 'get_array_of_samples' input
        # Convert back to 16-bit PCM for Pydub and common TTS/embedding models
        trimmed_audio_int16 = (trimmed_audio_array * (2**15 - 1)).astype(np.int16)
        final_audio_segment = AudioSegment(
            trimmed_audio_int16.tobytes(), 
            frame_rate=Config.SAMPLE_RATE, 
            sample_width=2, # 16-bit
            channels=1
        )

        # 4. Enforce maximum duration (cut from the end if too long)
        if len(final_audio_segment) > Config.MAX_AUDIO_DURATION_SECONDS * 1000: # pydub length is in ms
            original_duration_ms = len(final_audio_segment)
            final_audio_segment = final_audio_segment[:Config.MAX_AUDIO_DURATION_SECONDS * 1000]
            logger.warning(
                f"Audio sample trimmed from {original_duration_ms/1000:.2f}s "
                f"to max {Config.MAX_AUDIO_DURATION_SECONDS}s."
            )

        # 5. Save the processed audio
        # Exporting is a blocking I/O operation
        await loop.run_in_executor(
            None,
            # This is the crucial change: wrap the export call in a lambda
            lambda: final_audio_segment.export(str(output_audio_path), format="wav")
        )
        logger.info(f"Audio preprocessing finished. Processed audio saved to {output_audio_path}")

    except Exception as e:
        logger.exception(f"Error during audio preprocessing for {input_audio_path}: {e}")
        raise ValueError(f"Failed to preprocess audio: {e}")