# app/services/watermarking.py
import io
import numpy as np
from pydub import AudioSegment
from app.utils.exceptions import WatermarkingError

def add_watermark(audio_bytes: bytes, watermark_id: str = "REVOTIC_AI") -> bytes:
    """
    Adds a simple, inaudible watermark to the audio.
    
    NOTE: This is a basic implementation for demonstration. 
    A production system would use a more robust, spread-spectrum 
    or deep-learning-based watermarking technique.
    
    Args:
        audio_bytes: The synthesized audio data in bytes.
        watermark_id: A string identifier to embed.

    Returns:
        The watermarked audio data in bytes.
    """
    try:
        # Load the synthesized audio
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        samples = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate

        # 1. Generate a simple watermark signal (e.g., a very high-frequency tone)
        # We'll make its duration proportional to the length of the watermark_id
        duration_ms = len(watermark_id) * 50 # 50ms per character
        frequency = 20000 # 20kHz, generally inaudible to adults
        
        # Generate the sine wave for the watermark
        t = np.linspace(0, duration_ms / 1000, int(sample_rate * duration_ms / 1000), False)
        watermark_signal = np.sin(frequency * t * 2 * np.pi)
        
        # 2. Reduce its volume significantly to make it inaudible
        # We scale it to be a tiny fraction of the max amplitude of the original audio
        max_original_amplitude = np.max(np.abs(samples)) if np.any(samples) else 1.0
        watermark_signal = (watermark_signal * max_original_amplitude * 0.005).astype(np.int16)

        # 3. Append the watermark to the end of the audio
        watermarked_samples = np.concatenate([samples, watermark_signal])
        
        # 4. Convert back to an AudioSegment
        watermarked_audio = AudioSegment(
            watermarked_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )

        # Export to bytes in the same format as the input (e.g., mp3)
        output_buffer = io.BytesIO()
        watermarked_audio.export(output_buffer, format="mp3")
        return output_buffer.getvalue()

    except Exception as e:
        # If watermarking fails for any reason, we should not fail the entire request.
        # We'll log a warning and return the original audio.
        print(f"Warning: Watermarking failed: {e}. Returning original audio.")
        return audio_bytes
