"""
Everything this project claims, measured on the machine it is run on.

    python -m tools.device_report

Written for a remote Snapdragon laptop where nobody can watch the screen and
try again: every stage prints what it found as it goes, a failing stage says
why and the run continues, and the last thing printed is a summary of which
claims held on this machine and which did not.

Stages
    1  environment        what ONNX Runtime and the QNN plugin see here
    2  scanner            static rules on the shipped model
    3  compiler           Qualcomm's HTP compiler, run locally
    4  strict session     ORT with CPU fallback disabled: all on the NPU or nothing
    5  encoder latency    this project's converted encoder, on the NPU
    6  graph cache        cold versus warm session start
    7  vendor decoder     the piece this project does not run in its own app:
                          a pre-compiled Whisper decoder, loaded and timed here
    8  tests              the project's own suite, on this hardware

Nothing here is inferred from anything else. A number that could not be
measured is reported as not measured.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import traceback
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

MODEL = Path("models/whisper-tiny-qdq/encoder_model.onnx")

# Published, pre-compiled Whisper for Snapdragon X Elite. Used here only as a
# second, independent NPU workload: it proves a decoder runs on this chip and
# gives this machine's own number for it.
VENDOR_BUNDLE = (
    "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/"
    "whisper_tiny/releases/v0.63.0/"
    "whisper_tiny-precompiled_qnn_onnx-float-qualcomm_snapdragon_x_elite.zip"
)

CLOUD = {"encoder_ms": 17.8, "vendor_encoder_ms": 25.0, "vendor_decoder_ms": 2.60}

report: dict = {"stages": {}, "summary": {}}
_width = 78


def head(text: str) -> None:
    print("\n" + "=" * _width)
    print(text)
    print("=" * _width, flush=True)


def line(label: str, value) -> None:
    print(f"  {label:.<34} {value}", flush=True)


def ok(text: str) -> None:
    print(f"  [PASS] {text}", flush=True)


def fail(text: str) -> None:
    print(f"  [FAIL] {text}", flush=True)


def skip(text: str) -> None:
    print(f"  [ -- ] {text}", flush=True)


def stage(name: str):
    """Run a stage, record it, and never let it end the run."""
    def wrap(fn):
        head(f"{name}")
        started = time.time()
        try:
            value = fn()
            report["stages"][name] = {"ok": True, "seconds": round(time.time() - started, 1), **(value or {})}
        except Exception as exc:                       # noqa: BLE001 - a stage failing is a result
            report["stages"][name] = {
                "ok": False, "seconds": round(time.time() - started, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }
            fail(f"{type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
        return report["stages"][name]
    return wrap


# ── 1. environment ────────────────────────────────────────────────────────────

def environment() -> dict:
    import onnxruntime as ort

    from runtime.qnn_ep import compile_only_available, npu_available, register_plugin

    registered = register_plugin()
    try:
        import onnxruntime_qnn
        plugin = onnxruntime_qnn.__version__
    except ImportError:
        plugin = None

    devices = []
    if hasattr(ort, "get_ep_devices"):
        devices = [{"ep": d.ep_name, "type": str(d.device.type), "vendor": d.device.vendor}
                   for d in ort.get_ep_devices()]

    line("machine", f"{platform.machine()}  |  {platform.processor()[:40]}")
    line("python", platform.python_version())
    line("onnxruntime", ort.__version__)
    line("onnxruntime-qnn", plugin or "NOT INSTALLED")
    for d in devices:
        line(f"  device {d['ep']}", f"{d['type']}  ({d['vendor']})")

    npu = npu_available()
    if npu:
        ok("QNN EP exposes an NPU device on this machine")
    elif compile_only_available():
        fail("QNN EP is present but exposes no NPU - compile-only machine")
    else:
        fail("QNN EP is not available at all")

    return {"npu_available": bool(npu), "compile_only": bool(compile_only_available()),
            "onnxruntime": ort.__version__, "onnxruntime_qnn": plugin,
            "registered": bool(registered), "devices": devices,
            "machine": platform.machine(), "python": platform.python_version()}


# ── 2-3. scanner and compiler ─────────────────────────────────────────────────

def scanner() -> dict:
    from scanner.graph_analyzer import analyze_onnx_model

    r = analyze_onnx_model(str(MODEL))
    line("model", MODEL)
    line("NPU-eligible", f"{r.coverage_percent:.1f}%  ({r.supported_nodes}/{r.total_nodes} compute nodes)")
    line("quantization format", r.format_error["error"] if r.format_error else "static QDQ (correct for QNN)")
    fallback = {f["op_type"]: f["count"] for f in r.fallback_ops}
    line("ops that would fall back", fallback or "none")
    (ok if r.coverage_percent == 100.0 and not r.format_error else fail)(
        f"static analysis: {r.coverage_percent:.1f}% NPU-eligible")
    return {"coverage_percent": r.coverage_percent, "fallback_ops": fallback,
            "format_error": bool(r.format_error)}


def compiler() -> dict:
    from scanner.htp_compile import compile_check

    r = compile_check(str(MODEL))
    line("QNN SDK (local compiler)", r.get("qnn_version"))
    line("compile time", f"{r.get('compile_s')} s")
    line("NPU graphs", r.get("npu_graphs"))
    line("compute ops left on CPU", r.get("cpu_ops") or "none")
    if r.get("error"):
        fail(f"compiler error: {r['error']}")
    elif r.get("ok"):
        ok("compiled into ONE NPU graph with nothing left on CPU")
    else:
        fail(f"graph split into {r.get('npu_graphs')} NPU graphs")
    return {k: r.get(k) for k in ("ok", "npu_graphs", "cpu_ops", "compile_s", "qnn_version", "error")}


# ── 4-6. the shipped model on this machine ───────────────────────────────────

def _bench(session, feed, runs: int) -> dict:
    for _ in range(5):
        session.run(None, feed)
    times = []
    for _ in range(runs):
        started = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - started) * 1000)
    times.sort()
    return {"runs": len(times), "median_ms": round(statistics.median(times), 2),
            "p90_ms": round(times[int(len(times) * 0.9) - 1], 2),
            "min_ms": round(times[0], 2), "max_ms": round(times[-1], 2)}


def _feed_for(session) -> dict:
    feed = {}
    for spec in session.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in spec.shape]
        dtype = np.float16 if "float16" in spec.type else np.float32
        if "int" in spec.type:
            feed[spec.name] = np.zeros(shape, np.int32)
        else:
            feed[spec.name] = np.random.randn(*shape).astype(dtype) * 0.1
    return feed


def strict_and_latency(runs: int, npu: bool) -> dict:
    from runtime.qnn_ep import create_session

    cache = Path(".qnn_cache_device")
    out: dict = {}

    started = time.time()
    session = create_session(MODEL, {"htp_arch": "73"}, cache_dir=cache, strict=True)
    out["cold_load_s"] = round(time.time() - started, 2)
    out["providers"] = session.get_providers()
    ok("session built with CPU fallback DISABLED - QNN EP took every node")
    line("providers", ", ".join(out["providers"]))
    line("cold load (compile + cache write)", f"{out['cold_load_s']} s")

    started = time.time()
    session = create_session(MODEL, {"htp_arch": "73"}, cache_dir=cache, strict=True)
    out["warm_load_s"] = round(time.time() - started, 2)
    line("warm load (cached graph)", f"{out['warm_load_s']} s")
    if out["cold_load_s"] > 0:
        line("cache speedup", f"{out['cold_load_s'] / max(out['warm_load_s'], 0.01):.1f}x")

    if not npu:
        skip("latency not measured: no NPU here, so any timing would be a CPU timing")
        out["latency"] = {"skipped": "no NPU on this machine"}
        return out

    out["latency"] = _bench(session, _feed_for(session), runs)
    line("encoder latency (median)", f"{out['latency']['median_ms']} ms over {out['latency']['runs']} runs")
    line("           p90 / min / max", f"{out['latency']['p90_ms']} / {out['latency']['min_ms']} / {out['latency']['max_ms']} ms")
    line("cloud X Elite, same file", f"{CLOUD['encoder_ms']} ms")
    ok(f"encoder ran on the NPU at {out['latency']['median_ms']} ms median")
    return out


# ── 7. a decoder on this NPU ─────────────────────────────────────────────────

def vendor_decoder(runs: int, npu: bool, work: Path) -> dict:
    """
    The project's own app keeps the decoder on CPU. Whether a decoder can run
    on this chip at all is a separate question, and a pre-compiled one answers
    it here rather than by argument.

    Inputs are zeros: placement and timing are real, the tokens are not, and
    this is labelled as a per-step timing rather than a transcription.
    """
    from runtime.qnn_ep import create_session
    from speech.npu_decoder import DecoderShape, initial_state, step_inputs

    work.mkdir(parents=True, exist_ok=True)
    bundle = work / "vendor-whisper.zip"
    if not bundle.exists():
        print("  downloading the pre-compiled bundle (about 100 MB) ...", flush=True)
        started = time.time()
        partial = bundle.with_suffix(".part")
        with urlopen(VENDOR_BUNDLE, timeout=120) as response, open(partial, "wb") as handle:
            megabytes = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                megabytes += len(chunk) / 1e6
                if int(megabytes) % 20 == 0:
                    print(f"    {megabytes:.0f} MB ...", flush=True)
        partial.replace(bundle)
        line("download", f"{bundle.stat().st_size / 1e6:.0f} MB in {time.time() - started:.0f} s")
    folder = work / "vendor"
    if not folder.exists():
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(folder)
    inner = next(p for p in folder.iterdir() if p.is_dir())
    decoder, encoder = inner / "decoder.onnx", inner / "encoder.onnx"
    line("bundle", inner.name)

    out: dict = {}
    if not npu:
        skip("pre-compiled graphs need a real NPU: they are EPContext models, nothing to run here")
        return {"skipped": "no NPU on this machine"}

    shape = DecoderShape.from_model(decoder)
    line("decoder shape", f"{shape.layers} layers, {shape.heads} heads, {shape.cache_len}-slot cache")

    session = create_session(decoder, {"htp_arch": "73"}, cache_dir=Path(".qnn_cache_vendor"), strict=True)
    ok("decoder session built with CPU fallback DISABLED - every layer on the NPU")

    cross = {}
    for spec in session.get_inputs():
        if "cross" in spec.name:
            cross[spec.name] = np.zeros([d if isinstance(d, int) else 1 for d in spec.shape], np.float16)
    state = initial_state(shape)
    state = {k: v.astype(np.float16) for k, v in state.items()}
    feed = step_inputs(50258, 0, state, cross, shape)
    out["decoder"] = _bench(session, feed, runs)
    line("decoder latency per token", f"{out['decoder']['median_ms']} ms (median of {out['decoder']['runs']})")
    line("cloud X Elite, same bundle", f"{CLOUD['vendor_decoder_ms']} ms")

    try:
        enc_session = create_session(encoder, {"htp_arch": "73"}, cache_dir=Path(".qnn_cache_vendor"), strict=True)
        out["vendor_encoder"] = _bench(enc_session, _feed_for(enc_session), max(runs // 2, 10))
        line("vendor encoder (float16)", f"{out['vendor_encoder']['median_ms']} ms")
        line("cloud X Elite, same bundle", f"{CLOUD['vendor_encoder_ms']} ms")
    except Exception as exc:                            # noqa: BLE001
        skip(f"vendor encoder not timed: {exc}")

    ok("a decoder does run entirely on this NPU")
    return out


# ── 8. the project's own tests ───────────────────────────────────────────────

def tests() -> dict:
    import subprocess

    started = time.time()
    # tests/test_server.py needs fastapi and httpx, which are not worth
    # installing on a device being checked for NPU behaviour.
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--ignore=tests/test_server.py"],
                          capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-3:]
    for ln in tail:
        print(f"  {ln}", flush=True)
    passed = proc.returncode == 0
    (ok if passed else fail)(f"test suite {'passed' if passed else 'FAILED'} in {time.time() - started:.0f} s")
    return {"passed": passed, "returncode": proc.returncode, "tail": tail}


# ── the run ──────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=100, help="latency samples")
    parser.add_argument("--skip-vendor", action="store_true", help="skip the 100 MB download")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--json", type=Path, default=Path("device-report.json"))
    parser.add_argument("--work", type=Path, default=Path(".device-work"))
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    print("\nDragonn - full device report")
    print(f"started {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report["started"] = time.strftime("%Y-%m-%d %H:%M:%S")

    env = stage("1. Environment - what this machine exposes")(environment)
    npu = bool(env.get("npu_available"))

    if not MODEL.exists():
        head("Model")
        fail(f"{MODEL} is missing - the repository copy should have come with this checkout")
        report["summary"]["fatal"] = "model missing"
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 2

    stage("2. Scanner - static rules on the shipped model")(scanner)
    stage("3. Compiler - Qualcomm's HTP compiler, run locally")(compiler)
    stage("4-6. The shipped model on this machine")(lambda: strict_and_latency(args.runs, npu))
    if not args.skip_vendor:
        stage("7. A decoder on this NPU")(lambda: vendor_decoder(args.runs, npu, args.work))
    if not args.skip_tests:
        stage("8. The project's own test suite, on this hardware")(tests)

    # ── what held, and what did not ──────────────────────────────────────────
    head("SUMMARY - what this machine confirmed")
    s = report["stages"]
    checks = [
        ("NPU reachable from ONNX Runtime", npu),
        ("Shipped model: 100% NPU-eligible (static)", s.get("2. Scanner - static rules on the shipped model", {}).get("coverage_percent") == 100.0),
        ("Shipped model: one NPU graph (compiler)", s.get("3. Compiler - Qualcomm's HTP compiler, run locally", {}).get("ok") is True),
        ("Runs with CPU fallback disabled", s.get("4-6. The shipped model on this machine", {}).get("ok") is True),
        ("Encoder latency measured", bool(s.get("4-6. The shipped model on this machine", {}).get("latency", {}).get("median_ms"))),
        ("A decoder runs on this NPU", bool(s.get("7. A decoder on this NPU", {}).get("decoder"))),
        ("Core tests pass here (scanner, runtime, layering, decoder)", s.get("8. The project's own test suite, on this hardware", {}).get("passed") is True),
    ]
    for label, passed in checks:
        print(f"  {'[PASS]' if passed else '[ -- ]'} {label}", flush=True)
    report["summary"] = {label: bool(passed) for label, passed in checks}

    encoder = s.get("4-6. The shipped model on this machine", {}).get("latency", {})
    decoder = s.get("7. A decoder on this NPU", {}).get("decoder", {})
    if encoder.get("median_ms") or decoder.get("median_ms"):
        head("NUMBERS TO KEEP")
        if encoder.get("median_ms"):
            line("this project's encoder, on the NPU", f"{encoder['median_ms']} ms   (cloud: {CLOUD['encoder_ms']} ms)")
        if decoder.get("median_ms"):
            line("a decoder, per token, on the NPU", f"{decoder['median_ms']} ms   (cloud: {CLOUD['vendor_decoder_ms']} ms)")
        vendor_enc = s.get("7. A decoder on this NPU", {}).get("vendor_encoder", {})
        if vendor_enc.get("median_ms"):
            line("vendor float encoder, on the NPU", f"{vendor_enc['median_ms']} ms   (cloud: {CLOUD['vendor_encoder_ms']} ms)")
        if encoder.get("median_ms") and decoder.get("median_ms"):
            projected = encoder["median_ms"] + 50 * decoder["median_ms"]
            line("30 s clip, encoder + 50 tokens", f"{projected:.0f} ms (arithmetic over the two measurements)")

    failed = [name for name, data in s.items() if not data.get("ok")]
    if failed:
        head("STAGES THAT FAILED")
        for name in failed:
            print(f"  {name}: {s[name].get('error', 'see output above')}", flush=True)

    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
