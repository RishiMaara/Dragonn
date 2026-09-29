"""
Speech layer — Whisper transcription with the encoder on the NPU.

Depends on runtime only. The server and the evaluation tools depend on this.
"""

from speech.transcriber import WhisperTranscriber, load_audio, word_error_rate

__all__ = ["WhisperTranscriber", "load_audio", "word_error_rate"]
