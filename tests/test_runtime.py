"""
Runtime tests: AI Hub result parsing, WER, audio loading, model sizing, QNN EP
attachment. Each pins a bug that produced a wrong number at some point.
"""

import io
import platform

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

# Test models carry the IR version the real pipeline's models carry (torch's
# exporter writes 10). Newer onnx releases default to IR 14, which ONNX Runtime
# 1.26 refuses to load — that broke CI when it picked up onnx 1.23.
IR_VERSION = 10

from converter.quantize import _model_size_mb
from scripts.aihub_validate import reconcile, summarize_profile
from scripts.transcriber import load_audio, word_error_rate


# ── AI Hub profile parsing ────────────────────────────────────────────────────

def _profile(times_ms, units):
    return {
        "execution_summary": {
            "estimated_inference_time": min(times_ms) * 1000,
            "all_inference_times": [t * 1000 for t in times_ms],
            "estimated_inference_peak_memory": 44 * 1024**2,
            "first_load_time": 5_283_661,
            "warm_load_time": 526_612,
        },
        "execution_detail": [
            {"name": f"l{i}", "type": t, "compute_unit": u, "execution_time": 10}
            for i, (t, u) in enumerate(units)
        ],
    }


def test_latency_is_reported_as_median_not_ai_hubs_minimum():
    """AI Hub's headline number is the fastest run: 17.6 ms vs a 41.7 ms median."""
    s = summarize_profile(_profile([17.6, 40, 41, 42, 45], [("Conv", "NPU")]))
    assert s["inference_ms_min"] == 17.6
    assert s["inference_ms_median"] == 41.0
    assert s["inference_runs"] == 5


def test_graph_io_quantization_on_cpu_is_not_a_fallback():
    units = [("QuantizeLinear", "CPU"), ("Conv", "NPU"), ("DequantizeLinear", "CPU")]
    s = summarize_profile(_profile([40], units))
    r = reconcile({"effective_coverage_percent": 100.0, "predicted_fallback_ops": []}, s)
    assert r["verdict"].startswith("CONFIRMED")
    assert r["unexpected_fallback_types"] == []


def test_real_fallback_contradicting_scanner_is_flagged():
    units = [("LayerNorm", "CPU"), ("Conv", "NPU")]
    s = summarize_profile(_profile([40], units))
    r = reconcile({"effective_coverage_percent": 100.0, "predicted_fallback_ops": []}, s)
    assert r["verdict"].startswith("SCANNER OVER-PREDICTED")
    assert r["unexpected_fallback_types"] == ["LayerNorm"]


def test_empty_profile_does_not_crash():
    s = summarize_profile({})
    assert s["inference_ms_median"] is None and s["total_layers"] == 0


# ── WER and audio ─────────────────────────────────────────────────────────────

def test_word_error_rate():
    assert word_error_rate(["a b c"], ["a b c"])["wer_percent"] == 0.0
    assert word_error_rate(["a b c d"], ["a x c d"])["wer_percent"] == 25.0      # substitution
    assert word_error_rate(["a b c d"], ["a b c"])["wer_percent"] == 25.0        # deletion
    assert word_error_rate(["a b"], ["a b c d"])["wer_percent"] == 100.0         # insertions
    corpus = word_error_rate(["a b", "c d e f"], ["a x", "c d e f"])
    assert corpus["edits"] == 1 and corpus["ref_words"] == 6                     # corpus-level


def test_load_audio_resamples_and_downmixes_to_16k_mono():
    import soundfile as sf
    stereo_48k = np.zeros((48_000, 2), np.float32)
    buf = io.BytesIO()
    sf.write(buf, stereo_48k, 48_000, format="WAV")
    audio = load_audio(buf.getvalue())
    assert audio.ndim == 1 and abs(len(audio) - 16_000) <= 1


def test_load_audio_rejects_garbage():
    """The old dashboard posted these exact bytes and got a fake transcript back."""
    with pytest.raises(Exception):
        load_audio(b"dummy audio bytes")


# ── Model sizing ──────────────────────────────────────────────────────────────

def test_model_size_counts_external_data(tmp_path):
    """Measuring only the .onnx made a 3.9x compression read as a 270x expansion."""
    w = numpy_helper.from_array(np.ones((512, 512), np.float32), "w")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "w"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 512])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 512])], [w],
    )
    path = tmp_path / "m.onnx"
    onnx.save(helper.make_model(graph, ir_version=IR_VERSION), path, save_as_external_data=True,
              all_tensors_to_one_file=True, location="m.onnx.data", size_threshold=0)
    assert path.stat().st_size < 10_000
    assert _model_size_mb(path) >= 1.0          # 512*512*4 bytes = 1 MB of weights


# ── QNN EP attachment ─────────────────────────────────────────────────────────

# Scoped to these tests only — a module-level importorskip would silently skip
# every test in this file on machines without the plugin.
try:
    import onnxruntime_qnn  # noqa: F401
    HAS_QNN = True
except ImportError:
    HAS_QNN = False

needs_qnn = pytest.mark.skipif(not HAS_QNN, reason="onnxruntime-qnn not installed")


@needs_qnn
@pytest.mark.skipif(platform.machine().lower() in ("arm64", "aarch64"), reason="x64-only expectation")
def test_x64_reports_no_npu_even_with_qnn_installed():
    from scripts.qnn_ep import compile_only_available, npu_available
    assert compile_only_available()
    assert not npu_available()


def _add_model(path, doc=""):
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    model = helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 17)])
    model.doc_string = doc
    onnx.save(model, path)


@needs_qnn
def test_compiled_graph_cache_is_reused_and_invalidated_by_model_changes(tmp_path):
    """Cold start was 5.28 s vs 0.53 s warm on a real X Elite; the cache must never go stale."""
    from scripts.qnn_ep import create_session

    model, cache = tmp_path / "m.onnx", tmp_path / "cache"
    _add_model(model)
    create_session(model, {"htp_arch": "73"}, cache_dir=cache)
    first = sorted(p.name for p in cache.glob("*_ctx.onnx"))
    assert len(first) == 1

    session = create_session(model, {"htp_arch": "73"}, cache_dir=cache)   # served from cache
    assert "QNNExecutionProvider" in session.get_providers()
    assert sorted(p.name for p in cache.glob("*_ctx.onnx")) == first

    _add_model(model, doc="changed")                                          # different bytes
    create_session(model, {"htp_arch": "73"}, cache_dir=cache)
    assert len(list(cache.glob("*_ctx.onnx"))) == 2


@needs_qnn
def test_classic_provider_list_silently_drops_the_plugin_but_create_session_does_not(tmp_path):
    """The ORT 1.26 + onnxruntime-qnn 2.x trap that would have run the server on CPU."""
    import onnxruntime as ort
    from scripts.qnn_ep import create_session, htp_backend_path, register_plugin

    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"])], "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    path = tmp_path / "add.onnx"
    onnx.save(helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 17)]), path)
    register_plugin()

    so = ort.SessionOptions(); so.log_severity_level = 3
    classic = ort.InferenceSession(str(path), sess_options=so, providers=[
        ("QNNExecutionProvider", {"backend_path": htp_backend_path()}), "CPUExecutionProvider"])
    assert "QNNExecutionProvider" not in classic.get_providers()

    so = ort.SessionOptions(); so.log_severity_level = 3
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_file_path", str(tmp_path / "ctx.onnx"))
    session = create_session(path, {"htp_arch": "73"}, so)
    assert "QNNExecutionProvider" in session.get_providers()
