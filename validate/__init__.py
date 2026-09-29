"""
Validation layer — run a model on real Snapdragon hardware and reconcile what
the device did against what the scanner predicted.

    aihub       the full Whisper path: profile, accuracy, CPU baseline
    precompiled any already-compiled model (Qualcomm's own releases included)
"""

from validate.aihub import reconcile, summarize_profile, with_network_retry
from validate.precompiled import profile_precompiled, stage_for_hub

__all__ = ["reconcile", "summarize_profile", "with_network_retry",
           "profile_precompiled", "stage_for_hub"]
