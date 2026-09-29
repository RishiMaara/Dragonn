"""
Runtime layer — the bottom of the stack. Depends on nothing else in this
project, so every other package can rely on it without creating a cycle.

    qnn_ep  attaching QNN EP correctly, verifying it attached, caching the
            compiled HTP graph. The one door the NPU is reached through.
    audio   decode and resample, shared by the app and the calibration reader.
"""

from runtime.audio import CHUNK_SECONDS, SAMPLE_RATE, load_audio, resample
from runtime.qnn_ep import (
    compile_only_available,
    context_cache_path,
    create_session,
    htp_backend_path,
    npu_available,
    register_plugin,
)

__all__ = [
    "CHUNK_SECONDS", "SAMPLE_RATE", "load_audio", "resample",
    "compile_only_available", "context_cache_path", "create_session",
    "htp_backend_path", "npu_available", "register_plugin",
]
