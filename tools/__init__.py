"""
Command-line workflows. This package is a leaf: nothing in the project imports
from it, which is what keeps it from becoming the junk drawer it replaced.

    python -m tools.fetch_speech     download the evaluation speech
    python -m tools.run_pipeline     export -> quantize -> scan -> compile-check
    python -m tools.eval_wer         word error rate, locally and on a device
    python -m tools.model_zoo        naive vs this pipeline across five models
    python -m tools.profile_qnn      latency, with no silent CPU fallback
"""
