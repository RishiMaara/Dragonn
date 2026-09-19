"""
Hexagon Bridge — QNN Execution Provider setup
=============================================
The one place that attaches QNN EP to an ONNX Runtime session — correctly,
across both packagings — and fails loudly when it silently doesn't attach.

Packaging history, and the traps in each (verified on ORT 1.26 + onnxruntime-qnn 2.6):

  onnxruntime-qnn 1.x  QNN EP built into its own onnxruntime wheel. Listed by
                       ort.get_available_providers() out of the box.

  onnxruntime-qnn 2.x  A plugin EP loaded into stock onnxruntime (>= 1.24).
                       - Invisible until registered: get_available_providers()
                         does not list it, so the classic availability check
                         concludes "no QNN" and code falls back to CPU.
                       - Even after registering, the classic
                         providers=[("QNNExecutionProvider", {...})] argument
                         builds a session WITHOUT error that runs entirely on
                         CPU. session.get_providers() is the only tell.
                       - On x64 it registers a CPU-typed device: it can compile
                         HTP context binaries but cannot execute them.

So: register, attach via add_provider_for_devices, and verify.
"""

import logging
import platform
from pathlib import Path

logger = logging.getLogger("hexagon-bridge.qnn")

EP = "QNNExecutionProvider"
_registered = False


def register_plugin() -> bool:
    """Make QNN EP visible to this process. Returns True if QNN is available in any form."""
    global _registered
    import onnxruntime as ort

    if _registered or EP in ort.get_available_providers():
        return True
    try:
        import onnxruntime_qnn
    except ImportError:
        return False
    try:
        ort.register_execution_provider_library(EP, onnxruntime_qnn.get_library_path())
        _registered = True
        return True
    except Exception as e:
        logger.warning(f"Could not register the QNN plugin EP: {e}")
        return False


def _plugin_devices() -> list:
    import onnxruntime as ort

    if not hasattr(ort, "get_ep_devices"):
        return []
    return [d for d in ort.get_ep_devices() if d.ep_name == EP]


def npu_available() -> bool:
    """
    True only if QNN EP can EXECUTE on a Hexagon NPU here — not merely compile
    for one. An x64 machine with onnxruntime-qnn installed returns False.
    """
    if not register_plugin():
        return False

    devices = _plugin_devices()
    if devices:
        import onnxruntime as ort
        return any(d.device.type == ort.OrtHardwareDeviceType.NPU for d in devices)

    # 1.x in-box packaging has no device API; the HTP backend only executes on ARM64.
    return platform.machine().lower() in ("arm64", "aarch64")


def compile_only_available() -> bool:
    """True if QNN EP can at least compile HTP graphs here (e.g. x64 + onnxruntime-qnn)."""
    return register_plugin()


def htp_backend_path() -> str:
    try:
        import onnxruntime_qnn
        return onnxruntime_qnn.get_qnn_htp_path()
    except ImportError:
        return "QnnHtp.dll"  # 1.x: resolved next to the onnxruntime wheel's DLLs


def context_cache_path(model_path: str | Path, cache_dir: str | Path, provider_options: dict) -> Path:
    """
    Where the compiled HTP context for this exact model + QNN build + options lives.

    A context binary is only valid for the QNN SDK version and HTP target that
    produced it, and for the exact weights it was compiled from — so all of
    those go into the key. A stale cache is never reused; it simply misses.
    """
    import hashlib

    model_path = Path(model_path)
    digest = hashlib.sha256()
    for part in (model_path, model_path.with_name(model_path.name + ".data")):
        if part.exists():
            digest.update(part.read_bytes())
    try:
        from onnxruntime_qnn.build_and_package_info import qnn_version
    except ImportError:
        qnn_version = "inbox"
    digest.update(qnn_version.encode())
    digest.update(repr(sorted(provider_options.items())).encode())
    return Path(cache_dir) / f"{model_path.stem}.{digest.hexdigest()[:16]}_ctx.onnx"


def _open(model_path, options: dict, so):
    import onnxruntime as ort

    devices = _plugin_devices()
    if devices:
        so.add_provider_for_devices(devices, options)
        return ort.InferenceSession(str(model_path), sess_options=so)
    return ort.InferenceSession(
        str(model_path), sess_options=so, providers=[(EP, options), "CPUExecutionProvider"]
    )


def _verify(session, require_qnn: bool):
    attached = session.get_providers()
    if EP not in attached:
        message = (
            f"QNN EP did not attach — this session runs on {attached}. Any 'NPU' numbers "
            "from it would be CPU numbers."
        )
        if require_qnn:
            raise RuntimeError(message)
        logger.error(message)


def create_session(
    model_path: str | Path,
    provider_options: dict | None = None,
    sess_options=None,
    require_qnn: bool = True,
    cache_dir: str | Path | None = None,
):
    """
    Build an InferenceSession on QNN EP (HTP backend) with CPU fallback for any
    nodes QNN rejects, then verify QNN actually attached.

    Raises RuntimeError if QNN is missing from session.get_providers() and
    require_qnn is set — never hands back a CPU session labelled as NPU.

    cache_dir: keep the compiled HTP graph (an EP context model) there and load
    it on later starts instead of recompiling. On a real Snapdragon X Elite,
    whisper-tiny's cold load (which includes HTP graph finalization) measured
    5.28 s vs 0.53 s warm. Not combinable with caller-supplied sess_options.
    """
    import onnxruntime as ort

    if not register_plugin():
        raise RuntimeError(
            "QNN EP is not installed. On a Snapdragon device: pip install -r requirements-device.txt"
        )
    options = {"backend_path": htp_backend_path(), **(provider_options or {})}

    if cache_dir is None:
        session = _open(model_path, options, sess_options or ort.SessionOptions())
        _verify(session, require_qnn)
        return session

    if sess_options is not None:
        raise ValueError("cache_dir manages its own SessionOptions; don't pass sess_options too")

    ctx_path = context_cache_path(model_path, cache_dir, options)
    if ctx_path.exists():
        try:
            session = _open(ctx_path, options, ort.SessionOptions())
            _verify(session, require_qnn)
            logger.info(f"Loaded cached HTP context {ctx_path.name} — graph compilation skipped")
            return session
        except Exception as e:
            logger.warning(f"Cached HTP context unusable ({e}); recompiling")
            ctx_path.unlink(missing_ok=True)

    ctx_path.parent.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_file_path", str(ctx_path))
    session = _open(model_path, options, so)
    _verify(session, require_qnn)
    logger.info(f"Compiled HTP graph and cached it as {ctx_path.name}")
    return session
