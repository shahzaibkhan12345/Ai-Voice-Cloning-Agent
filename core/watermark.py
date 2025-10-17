import os
import asyncio
import numpy as np
import soundfile as sf
import hashlib # For creating a robust unique ID for the watermark
from pathlib import Path
from typing import Optional, Tuple

from config import Config
from api.utils import get_logger

logger = get_logger(__name__)

async def embed_watermark(
    input_audio_path: Path,
    output_audio_path: Path,
    unique_id: str
) -> Optional[Path]:
    """
    Embeds an inaudible watermark into an audio file.
    This implementation uses a simple frequency-based embedding in the ultrasonic range.
    The unique_id is encoded as a binary sequence by subtle amplitude modulation
    of specific high frequencies.

    Args:
        input_audio_path: Path to the audio file to watermark (e.g., WAV).
        output_audio_path: Path where the watermarked audio file will be saved.
        unique_id: A string (e.g., request_id or user_id) to embed as the watermark.

    Returns:
        Path to the watermarked audio file if successful, None otherwise.
    """
    logger.info(f"Embedding watermark for unique ID '{unique_id}' into {input_audio_path}...")

    if not input_audio_path.exists():
        logger.error(f"Input audio file for watermarking not found: {input_audio_path}")
        return None

    try:
        # Load audio data asynchronously
        loop = asyncio.get_running_loop()
        audio_data, samplerate = await loop.run_in_executor(
            None, sf.read, str(input_audio_path)
        )
        
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1) # Convert to mono if stereo

        # Normalize audio to float32 range [-1, 1]
        audio_data = audio_data.astype(np.float32)
        if audio_data.max() > 1.0 or audio_data.min() < -1.0:
             audio_data = audio_data / np.max(np.abs(audio_data)) # Ensure normalized

        # Convert unique_id to a binary string representation
        # Using SHA256 hash for consistency and fixed length
        hashed_id = hashlib.sha256(unique_id.encode('utf-8')).hexdigest()
        binary_watermark = bin(int(hashed_id, 16))[2:].zfill(256) # 256 bits for SHA256

        # Define watermark frequencies
        # These should be inaudible (e.g., above 18-20 kHz for humans)
        # Ensure they are below Nyquist frequency (samplerate / 2)
        nyquist_freq = samplerate / 2
        
        # We need at least two distinct frequencies to encode '0' and '1'
        # Or, we can use a single frequency and modulate its amplitude/presence
        # For simplicity, let's slightly modulate an ultrasonic carrier.
        
        # A more robust approach would be DSSS (Direct Sequence Spread Spectrum) or
        # phase modulation, but for a simple, detectable watermark,
        # slight amplitude modulation of a carrier can work.

        # Let's try to embed each bit by slightly altering a high frequency band's amplitude
        # Divide the ultrasonic band into 'bit' slots
        
        # For this example, let's use a simpler, more noticeable (but still high frequency)
        # modulation for demonstration. For truly inaudible, this would need more
        # sophisticated DSSS or spread spectrum techniques.
        # Let's pick a high frequency, and modulate its presence/absence or amplitude.

        # Simple approach: embed bit by slightly increasing/decreasing amplitude in a high-freq band
        # This is a very basic method and actual robust watermarking is complex.
        # For production-grade, a dedicated library or service would be ideal.

        # For demonstration: We'll embed the first N bits of the hash
        # Let's say we embed 10 bits.
        num_bits_to_embed = 10 
        carrier_freq = Config.WATERMARK_FREQ_LOW + ((Config.WATERMARK_FREQ_HIGH - Config.WATERMARK_FREQ_LOW) / 2) # Mid-point ultrasonic
        amplitude_scale = Config.WATERMARK_AMPLITUDE_SCALE # From config, very small

        if carrier_freq >= nyquist_freq:
            logger.warning(
                f"Watermark carrier frequency {carrier_freq}Hz is too high for samplerate {samplerate}Hz. "
                "Skipping watermarking or adjusting frequency."
            )
            # Adjust frequency to be just below Nyquist
            carrier_freq = nyquist_freq * 0.95 
            logger.warning(f"Adjusted carrier frequency to {carrier_freq:.2f}Hz.")

        # Create carrier wave (sine wave)
        t = np.linspace(0., len(audio_data) / samplerate, len(audio_data), endpoint=False, dtype=np.float32)
        carrier_wave = np.sin(2 * np.pi * carrier_freq * t)

        # Modulate carrier wave based on bits of the unique_id (hash)
        # Divide the audio into segments, and for each segment embed a bit.
        segment_length_samples = len(audio_data) // num_bits_to_embed
        
        watermark_signal = np.zeros_like(audio_data)
        for i in range(min(num_bits_to_embed, len(binary_watermark))):
            bit = int(binary_watermark[i])
            start_idx = i * segment_length_samples
            end_idx = start_idx + segment_length_samples
            
            if bit == 1:
                # Add a small amount of the carrier wave for a '1' bit
                watermark_signal[start_idx:end_idx] += carrier_wave[start_idx:end_idx] * amplitude_scale
            # For '0', we add nothing or a different (even smaller) modulation

        # Add the watermark signal to the original audio
        watermarked_audio_data = audio_data + watermark_signal

        # Clip values to ensure they remain within valid audio range [-1, 1]
        watermarked_audio_data = np.clip(watermarked_audio_data, -1.0, 1.0)

        # Save the watermarked audio
        await loop.run_in_executor(
            None, sf.write, str(output_audio_path), watermarked_audio_data, samplerate
        )
        logger.info(f"Watermark embedded successfully. Saved to {output_audio_path}")
        return output_audio_path

    except Exception as e:
        logger.exception(f"Error embedding watermark into {input_audio_path} for ID '{unique_id}': {e}")
        return None