# app/utils/exceptions.py

class VoiceCloningError(Exception):
    """Base exception for voice cloning errors."""
    pass

class AudioProcessingError(VoiceCloningError):
    """Raised when audio preprocessing fails."""
    pass

class SynthesisError(VoiceCloningError):
    """Raised when the voice synthesis API call fails."""
    pass

class WatermarkingError(VoiceCloningError):
    """Raised when audio watermarking fails."""
    pass