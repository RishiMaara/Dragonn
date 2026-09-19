"""
Hexagon Bridge — Real-Hardware Validation via Qualcomm AI Hub
=============================================================
Runs the quantized model on a real, cloud-hosted Snapdragon X Elite through
ONNX Runtime + QNN EP, then checks the scanner's static prediction against what
the device actually did.

Three questions, each answered by the device rather than by static analysis:

  1. PLACEMENT  — which layers actually ran on the Hexagon NPU vs CPU?
                  Compared against the scanner's predicted coverage.
  2. ACCURACY   — does the a16w8 model compute the right answer ON HTP?
                  Device output vs local FP32, and vs the same QDQ model on
                  local CPU (isolates HTP numeric divergence from quantization
                  error).
  3. COST       — on-device latency, peak memory, and first-load time (which
                  includes HTP graph finalization — the cold-start cost that
                  EP context caching exists to remove).

The model is uploaded to Qualcomm AI Hub. Jobs are private to your account
unless you share them.

Setup (once):
    1. Create a free account at https://aihub.qualcomm.com
    2. Copy your API token from Account -> Settings
    3. qai-hub configure --api_token <YOUR_TOKEN>

Usage:
    # Everything local — no upload, no account needed. Validates the harness.
    python -m scripts.aihub_validate --dry-run

    # Real run on a cloud Snapdragon X Elite
    python -m scripts.aihub_validate

    # See which Snapdragon PC devices your account can target
    python -m scripts.aihub_validate --list-devices
"""

import argparse
import json
import logging
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger("hexagon-bridge.aihub")

DEFAULT_DEVICE = "Snapdragon X Elite CRD"
DEFAULT_MODEL = "./models/whisper-tiny-qdq/encoder_model.onnx"
DEFAULT_FP32 = "./models/whisper-tiny-onnx/encoder_model.onnx"
DEFAULT_REPORT = "./models/reports/aihub_validation.json"

# Calibration draws synthetic samples from indices 0..N. The test input comes
# from far outside that range so the accuracy check is on held-out data.
HELD_OUT_INDEX = 10_000

# Cosine similarity thresholds for the accuracy verdicts.
COSINE_PASS = 0.99
COSINE_WARN = 0.95

# Always route ONNX through QNN EP. Without this, AI Hub is free to pick another
# execution provider and the placement numbers would say nothing about Hexagon.
QNN_OPTIONS = "--onnx_execution_providers qnn --compute_unit all"

# Same device, CPU only — the baseline the NPU numbers are compared against.
CPU_OPTIONS = "--compute_unit cpu"

SETUP_HELP = """
Qualcomm AI Hub is not configured on this machine.

  1. Create a free account:  https://aihub.qualcomm.com
  2. Copy your API token:    Account -> Settings -> API Token
  3. Configure the client:   qai-hub configure --api_token <YOUR_TOKEN>

Then re-run:  python -m scripts.aihub_validate

(--dry-run works without an account and validates everything except the device.)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _rel_l2(ref: np.ndarray, x: np.ndarray) -> float:
    denom = np.linalg.norm(ref)
    return float(np.linalg.norm(ref - x) / denom * 100) if denom > 0 else 0.0


def _compare(ref: np.ndarray, x: np.ndarray) -> dict:
    cos = _cosine(ref, x)
    if cos >= COSINE_PASS:
        verdict = "PASS"
    elif cos >= COSINE_WARN:
        verdict = "WARN"
    else:
        verdict = "FAIL"
    return {
        "cosine": round(cos, 6),
        "rel_l2_percent": round(_rel_l2(ref, x), 3),
        "max_abs_err": round(float(np.abs(ref - x).max()), 5),
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Local preparation
# ─────────────────────────────────────────────────────────────────────────────

def ensure_inline_weights(model_path: Path) -> Path:
    """
    Return a path to a single-file version of the model.

    AI Hub uploads one .onnx file; weights in a separate .onnx.data sidecar would
    be left behind. Models under the 2 GB protobuf limit are re-saved inline.
    """
    import onnx

    header = onnx.load(str(model_path), load_external_data=False)
    has_external = any(
        entry.key == "location"
        for init in header.graph.initializer
        for entry in init.external_data
    )
    if not has_external:
        return model_path

    full = onnx.load(str(model_path))  # pulls the sidecar in
    if full.ByteSize() >= 2 * 1024**3:
        raise ValueError(
            f"{model_path.name} exceeds the 2 GB single-file protobuf limit. "
            "Upload it as an AI Hub model directory instead."
        )

    inline_path = Path(tempfile.mkdtemp(prefix="hexbridge_aihub_")) / model_path.name
    onnx.save(full, str(inline_path), save_as_external_data=False)
    logger.info(f"Inlined external weights for upload -> {inline_path}")
    return inline_path


def build_test_input(model_path: Path, model_id: str, audio_dir: str | None) -> dict:
    """
    One held-out input, produced by the same feature path calibration uses —
    audio pushed through Whisper's real WhisperFeatureExtractor.
    """
    from converter.quantize import WhisperCalibrationDataReader

    reader = WhisperCalibrationDataReader(
        str(model_path), num_samples=1, model_id=model_id, audio_dir=audio_dir
    )
    if reader._feature_extractor is None:
        raise RuntimeError(
            "WhisperFeatureExtractor unavailable — install transformers. A random "
            "input would make the accuracy comparison meaningless."
        )

    feeds = {}
    for name, spec in reader.input_specs.items():
        if "input_features" not in name:
            raise RuntimeError(f"Unexpected model input '{name}' — expected a Whisper encoder")
        feeds[name] = reader._make_log_mel(spec, HELD_OUT_INDEX)
    return feeds


def run_local(model_path: Path, feeds: dict) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)[0]


def scanner_prediction(model_path: Path) -> dict:
    from scanner.graph_analyzer import analyze_onnx_model

    report = analyze_onnx_model(model_path)
    return {
        "effective_coverage_percent": report.effective_coverage_percent,
        "node_coverage_percent": report.coverage_percent,
        "compute_nodes": report.total_nodes,
        "format_error": report.format_error,
        "predicted_fallback_ops": [f["op_type"] for f in report.fallback_ops],
        "partial_ops": sorted({
            op for op, count in report.op_type_counts.items()
            if _support_level(op) == "partial"
        }),
    }


def _support_level(op_type: str) -> str:
    from scanner.op_registry import is_supported
    return is_supported(op_type)[1]


# ─────────────────────────────────────────────────────────────────────────────
# AI Hub
# ─────────────────────────────────────────────────────────────────────────────

def import_hub():
    try:
        import qai_hub as hub
        return hub
    except ImportError:
        logger.error("qai-hub is not installed. Run: pip install qai-hub")
        sys.exit(1)


def check_auth(hub) -> bool:
    try:
        hub.get_devices(name=DEFAULT_DEVICE)
        return True
    except Exception as e:
        logger.debug(f"AI Hub auth check failed: {e}")
        print(SETUP_HELP)
        return False


def list_pc_devices(hub) -> None:
    """Print the Snapdragon compute (Windows) devices this account can target."""
    devices = hub.get_devices()
    pcs = [
        d for d in devices
        if "windows" in (d.os or "").lower()
        or "snapdragon x" in d.name.lower()
        or any("snapdragon-x" in a for a in d.attributes)
    ]
    if not pcs:
        print("No Snapdragon PC devices visible to this account.")
        return
    print(f"\nSnapdragon PC devices available ({len(pcs)}):")
    for d in sorted({(d.name, d.os) for d in pcs}):
        print(f"  {d[0]:40}  os={d[1]}")
    print(f"\nUse one with:  python -m scripts.aihub_validate --device \"<name>\"")


def resolve_device(hub, name: str):
    if hub.get_devices(name=name):
        return hub.Device(name)
    logger.error(f"Device '{name}' is not available to this account.")
    list_pc_devices(hub)
    sys.exit(1)


def _with_network_retry(fn, job, label: str, attempts: int = 20, delay_s: int = 30):
    """
    Call fn(), riding out network drops.

    The job runs on Qualcomm's device regardless of our connection — only our
    polling fails. The client's own retries give up after ~5 quick attempts, which
    a flaky hotspot or NAT64 DNS hiccup exhausts in under a minute, killing the
    run and discarding a result the device already produced. So retry slowly,
    and on final failure say how to re-attach instead of resubmitting.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts:
                if job is None:   # e.g. an upload: nothing on the server to re-attach to
                    raise RuntimeError(f"{label}: network kept failing ({e.__class__.__name__})") from e
                flag = "--inference-job" if "nference" in label else "--profile-job"
                raise RuntimeError(
                    f"{label}: network kept failing ({e.__class__.__name__}). The job is "
                    f"still running on the device — re-attach instead of resubmitting:\n"
                    f"  python -m scripts.aihub_validate {flag} {job.job_id}"
                ) from e
            logger.warning(
                f"  {label}: network error ({e.__class__.__name__}) — job keeps running "
                f"on the device. Retry {attempt}/{attempts - 1} in {delay_s}s..."
            )
            time.sleep(delay_s)


def _upload(hub, model_path, name: str):
    """Upload a model, retrying through network drops — re-sending is always safe."""
    path = str(ensure_inline_weights(Path(model_path)))
    return _with_network_retry(lambda: hub.upload_model(path, name=name), None, f"Upload {name}")


def _wait(job, label: str):
    logger.info(f"  {label}: {job.url}")
    status = _with_network_retry(job.wait, job, label)
    if not getattr(status, "success", False):
        message = getattr(status, "message", "") or str(status)
        raise RuntimeError(f"{label} failed: {message}\n  Logs: {job.url}")
    return status


def summarize_profile(profile: dict) -> dict:
    """
    Turn AI Hub's raw profile into placement and cost numbers.

    Placement is reported two ways. Layer count is what the scanner predicts;
    time share is what the battery feels — one heavy op on CPU can dominate a
    graph that is 99% NPU by count.
    """
    summary = profile.get("execution_summary", {})
    layers = profile.get("execution_detail", [])

    by_count = Counter(layer.get("compute_unit", "UNSPECIFIED") for layer in layers)
    time_by_unit = defaultdict(int)
    cpu_types = Counter()
    for layer in layers:
        unit = layer.get("compute_unit", "UNSPECIFIED")
        time_by_unit[unit] += layer.get("execution_time", 0) or 0
        if unit != "NPU":
            cpu_types[layer.get("type", "?")] += 1

    total_layers = sum(by_count.values())
    total_time = sum(time_by_unit.values())

    def us_to_ms(value):
        return round(value / 1000, 3) if isinstance(value, (int, float)) and value > 0 else None

    def to_mb(value):
        return round(value / 1024**2, 2) if isinstance(value, (int, float)) and value > 0 else None

    # AI Hub's "estimated_inference_time" is the FASTEST of its runs, not a typical
    # one. On whisper-tiny / X Elite it read 17.6 ms while the median of 100 runs
    # was 41.7 ms — the distribution is bimodal. Report the distribution.
    runs = [t for t in summary.get("all_inference_times", []) if isinstance(t, (int, float)) and t > 0]
    runs_ms = np.array(runs) / 1000 if runs else None

    return {
        "total_layers": total_layers,
        "layers_by_unit": dict(by_count),
        "npu_layer_percent": round(by_count.get("NPU", 0) / total_layers * 100, 1) if total_layers else 0.0,
        "npu_time_percent": round(time_by_unit.get("NPU", 0) / total_time * 100, 1) if total_time else None,
        "non_npu_layer_types": dict(cpu_types.most_common()),
        "inference_ms_median": round(float(np.median(runs_ms)), 2) if runs_ms is not None else None,
        "inference_ms_p90": round(float(np.percentile(runs_ms, 90)), 2) if runs_ms is not None else None,
        "inference_ms_min": us_to_ms(summary.get("estimated_inference_time")),
        "inference_runs": len(runs),
        "inference_peak_memory_mb": to_mb(summary.get("estimated_inference_peak_memory")),
        # First load includes HTP graph finalization. This is the cold-start tax
        # an EP context binary removes; warm load shows what caching buys.
        "first_load_ms": us_to_ms(summary.get("first_load_time")),
        "warm_load_ms": us_to_ms(summary.get("warm_load_time")),
    }


def reconcile(prediction: dict, placement: dict) -> dict:
    """Where the scanner and the device disagree — each mismatch is a registry fix."""
    predicted = prediction["effective_coverage_percent"]
    actual = placement["npu_layer_percent"]
    # Graph-boundary QuantizeLinear/DequantizeLinear run on CPU by design: ORT's
    # QNN EP offloads graph I/O (de)quantization (offload_graph_io_quantization=1)
    # so float data can enter and leave the NPU. They are not fallbacks, and the
    # scanner deliberately excludes QDQ scaffolding — don't report them as misses.
    fell_back = {
        op: n for op, n in placement["non_npu_layer_types"].items()
        if op not in ("QuantizeLinear", "DequantizeLinear")
    }

    if actual == 0:
        verdict = (
            "NOTHING RAN ON THE NPU. QNN EP claimed no part of the graph — check the "
            "profile job logs for the partitioner's rejection reason."
        )
    elif not fell_back:
        verdict = (
            "CONFIRMED — every compute layer ran on the NPU, as the scanner predicted "
            "(graph I/O quantize/dequantize run on CPU by design)."
        )
    elif predicted >= 99.9:
        verdict = (
            "SCANNER OVER-PREDICTED — it claimed full coverage but these layer types ran "
            "off-NPU. Downgrade them in scanner/op_registry.py."
        )
    else:
        verdict = "PARTIAL — compare the fallback types below with the scanner's list."

    return {
        "predicted_npu_percent": predicted,
        "device_npu_layer_percent": actual,
        "device_npu_time_percent": placement["npu_time_percent"],
        "unexpected_fallback_types": sorted(
            set(fell_back) - set(prediction["predicted_fallback_ops"])
        ),
        "verdict": verdict,
    }


def run_cpu_baseline(hub, args, model_path: Path, fp32_path: Path) -> None:
    """
    The before/after receipt — one device, three configurations:

      FP32 on CPU   what a user runs today without this project
      QDQ  on CPU   the silent-fallback case: quantized, NPU never engaged
      QDQ  on NPU   this project

    Every job is submitted before any is awaited, so they run concurrently. A
    failed job is recorded rather than aborting the others.
    """
    if args.profile_job:
        npu_job = hub.get_job(args.profile_job)
        device, qdq_model = npu_job.device, npu_job.model
        logger.info(f"Reusing NPU profile job {args.profile_job} on {device.name}")
    else:
        device = resolve_device(hub, args.device)
        qdq_model = _upload(hub, model_path, f"hexbridge-{model_path.stem}")
        npu_job = hub.submit_profile_job(
            model=qdq_model, device=device, name="hexbridge-baseline-qdq-npu", options=QNN_OPTIONS
        )

    logger.info(f"Uploading FP32 reference model ({fp32_path})...")
    fp32_model = _upload(hub, fp32_path, f"hexbridge-{fp32_path.stem}-fp32")
    jobs = [
        ("FP32 on CPU", "today, without this project", hub.submit_profile_job(
            model=fp32_model, device=device, name="hexbridge-baseline-fp32-cpu", options=CPU_OPTIONS)),
        ("QDQ on CPU", "silent fallback", hub.submit_profile_job(
            model=qdq_model, device=device, name="hexbridge-baseline-qdq-cpu", options=CPU_OPTIONS)),
        ("QDQ on NPU", "this project", npu_job),
    ]

    results = {}
    for label, meaning, job in jobs:
        try:
            _wait(job, label)
            summary = summarize_profile(_with_network_retry(job.download_profile, job, label))
        except Exception as e:
            logger.error(f"{label} failed: {e}")
            summary = {"error": str(e)}
        summary.update({"meaning": meaning, "job": job.url})
        results[label] = summary

    base = results["FP32 on CPU"].get("inference_ms_median")
    for summary in results.values():
        median = summary.get("inference_ms_median")
        summary["speedup_vs_fp32_cpu"] = round(base / median, 2) if base and median else None

    report = {"device": device.name, "configs": results}
    out = Path(args.json).with_name("aihub_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    bar = "=" * 86
    print(f"\n{bar}\n  SAME DEVICE, CPU vs NPU — {device.name}\n{bar}")
    print(f"  {'config':12} {'what it is':30} {'median':>8} {'p90':>8} {'min':>8} {'peak MB':>8} {'speedup':>8}")
    print(f"  {'-' * 84}")
    for label, s in results.items():
        if "error" in s:
            print(f"  {label:12} {s['meaning']:30} FAILED — {s['error'].splitlines()[0][:60]}")
            continue
        speedup = f"{s['speedup_vs_fp32_cpu']}x" if s["speedup_vs_fp32_cpu"] else "n/a"
        print(f"  {label:12} {s['meaning']:30} {s['inference_ms_median']:>7}ms {s['inference_ms_p90']:>7}ms "
              f"{s['inference_ms_min']:>7}ms {s['inference_peak_memory_mb']!s:>8} {speedup:>8}")
    print(f"\n  Latency is the median of each job's runs; speedup is FP32-CPU median / config median.")
    for label, s in results.items():
        print(f"  {label:12} {s['job']}")
    print(f"\n  Report: {out}\n{bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Hexagon Bridge — validate on a real Snapdragon X via Qualcomm AI Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1],
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Quantized QDQ model to validate")
    parser.add_argument("--fp32", default=DEFAULT_FP32, help="FP32 reference model for accuracy")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help=f"AI Hub device (default: {DEFAULT_DEVICE})")
    parser.add_argument("--model-id", default=None, help="HF id for the feature extractor (default: config.json)")
    parser.add_argument("--audio-dir", default=None, help="Real speech for the test input (recommended)")
    parser.add_argument("--json", default=DEFAULT_REPORT, help="Where to write the validation report")
    parser.add_argument("--list-devices", action="store_true", help="List Snapdragon PC devices and exit")
    parser.add_argument("--skip-inference", action="store_true", help="Profile only; skip the accuracy job")
    parser.add_argument("--dry-run", action="store_true", help="Run every local step; submit nothing")
    parser.add_argument("--profile-job", default=None, help="Re-attach to an existing profile job ID instead of submitting")
    parser.add_argument("--inference-job", default=None, help="Re-attach to an existing inference job ID instead of submitting")
    parser.add_argument("--cpu-baseline", action="store_true",
                        help="Profile FP32 and QDQ on the device CPU and compare with the NPU "
                             "(add --profile-job to reuse an existing NPU run)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_devices:
        hub = import_hub()
        if check_auth(hub):
            list_pc_devices(hub)
        return

    model_path = Path(args.model)
    fp32_path = Path(args.fp32)
    for p in (model_path, fp32_path):
        if not p.exists():
            logger.error(f"Not found: {p}  (run the pipeline first: python -m scripts.run_pipeline)")
            sys.exit(1)

    if args.cpu_baseline:
        hub = import_hub()
        if not check_auth(hub):
            sys.exit(1)
        run_cpu_baseline(hub, args, model_path, fp32_path)
        return

    model_id = args.model_id
    if model_id is None:
        config = model_path.parent / "config.json"
        model_id = "openai/whisper-tiny"
        if config.exists():
            model_id = json.loads(config.read_text()).get("model_id", model_id)

    # ── Local: prediction, input, reference outputs ──
    logger.info("Step 1/4: Scanner prediction (static analysis)")
    prediction = scanner_prediction(model_path)
    if prediction["format_error"]:
        logger.error(
            "Scanner reports a quantization format error — this model cannot run on the "
            "NPU. Fix that before spending device time on it."
        )
        sys.exit(1)

    logger.info("Step 2/4: Held-out test input + local reference outputs")
    feeds = build_test_input(model_path, model_id, args.audio_dir)
    ref_fp32 = run_local(fp32_path, feeds)
    ref_qdq_cpu = run_local(model_path, feeds)
    quantization_error = _compare(ref_fp32, ref_qdq_cpu)

    report = {
        "model": str(model_path.resolve()),
        "device": args.device,
        "options": QNN_OPTIONS,
        "scanner_prediction": prediction,
        "local": {"qdq_cpu_vs_fp32": quantization_error},
    }

    if args.dry_run:
        report["dry_run"] = True
        _write_and_print(report, args.json)
        print("\nDry run complete. Nothing was uploaded. Drop --dry-run to run on the device.")
        return

    # ── Device ──
    hub = import_hub()
    if not check_auth(hub):
        sys.exit(1)
    started = time.time()
    if args.profile_job:
        # Re-attach: reuse the model and device the original job was submitted with.
        profile_job = hub.get_job(args.profile_job)
        uploaded, device = profile_job.model, profile_job.device
        report["device"] = device.name
        logger.info(f"Step 3/4: Re-attaching to profile job {args.profile_job} on {device.name}")
    else:
        device = resolve_device(hub, args.device)
        upload_path = ensure_inline_weights(model_path)
        logger.info(f"Step 3/4: Profiling on {args.device} (ONNX Runtime + QNN EP)")
        uploaded = _upload(hub, upload_path, f"hexbridge-{model_path.stem}")
        profile_job = hub.submit_profile_job(
            model=uploaded, device=device, name=f"hexbridge-profile-{model_path.stem}",
            options=QNN_OPTIONS,
        )
    report["profile_job"] = profile_job.url
    _wait(profile_job, "Profile job")
    placement = summarize_profile(
        _with_network_retry(profile_job.download_profile, profile_job, "Profile job")
    )
    report["device_placement"] = placement
    report["reconciliation"] = reconcile(prediction, placement)

    if not args.skip_inference:
        logger.info(f"Step 4/4: Inference on {report['device']} — does HTP compute the right answer?")
        if args.inference_job:
            inference_job = hub.get_job(args.inference_job)
        else:
            inference_job = hub.submit_inference_job(
                model=uploaded, device=device, name=f"hexbridge-infer-{model_path.stem}",
                inputs={name: [value] for name, value in feeds.items()},
                options=QNN_OPTIONS,
            )
        report["inference_job"] = inference_job.url
        _wait(inference_job, "Inference job")
        outputs = _with_network_retry(
            inference_job.download_output_data, inference_job, "Inference job"
        )
        device_out = np.asarray(next(iter(outputs.values()))[0]).reshape(ref_fp32.shape)
        report["device_accuracy"] = {
            # HTP vs the same model on CPU: pure hardware/runtime divergence.
            "device_vs_qdq_cpu": _compare(ref_qdq_cpu, device_out),
            # HTP vs FP32: what a user actually gets.
            "device_vs_fp32": _compare(ref_fp32, device_out),
        }

    report["wall_time_s"] = round(time.time() - started, 1)
    _write_and_print(report, args.json)


def _write_and_print(report: dict, json_path: str) -> None:
    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    bar = "=" * 70
    pred = report["scanner_prediction"]
    print(f"\n{bar}\n  HEXAGON BRIDGE — REAL-HARDWARE VALIDATION\n{bar}")
    print(f"  Device:     {report['device']}{'   (DRY RUN — not contacted)' if report.get('dry_run') else ''}")
    print(f"  Scanner:    {pred['effective_coverage_percent']:.1f}% NPU predicted "
          f"({pred['compute_nodes']} compute nodes; partial: {', '.join(pred['partial_ops']) or 'none'})")

    q = report["local"]["qdq_cpu_vs_fp32"]
    print(f"  Quant err:  cosine {q['cosine']:.5f}  rel-L2 {q['rel_l2_percent']:.2f}%  [{q['verdict']}]  (local CPU)")

    if "device_placement" in report:
        p, r = report["device_placement"], report["reconciliation"]
        time_share = f"{p['npu_time_percent']}%" if p["npu_time_percent"] is not None else "n/a"
        print(f"\n  ON DEVICE")
        print(f"  NPU layers: {p['npu_layer_percent']}%  ({p['layers_by_unit']})  |  NPU time share: {time_share}")
        if p["non_npu_layer_types"]:
            print(f"  Off-NPU:    {p['non_npu_layer_types']}")
        print(f"  Latency:    median {p['inference_ms_median']} ms   p90 {p['inference_ms_p90']} ms   "
              f"min {p['inference_ms_min']} ms (AI Hub's headline)   over {p['inference_runs']} runs")
        print(f"  Memory:     peak {p['inference_peak_memory_mb']} MB")
        print(f"  Load:       first {p['first_load_ms']} ms  (includes HTP finalization)   warm {p['warm_load_ms']} ms")
        print(f"\n  {r['verdict']}")
        if r["unexpected_fallback_types"]:
            print(f"  Not predicted by scanner: {', '.join(r['unexpected_fallback_types'])}")

    if "device_accuracy" in report:
        for label, key in (("HTP vs QDQ-CPU", "device_vs_qdq_cpu"), ("HTP vs FP32", "device_vs_fp32")):
            a = report["device_accuracy"][key]
            print(f"  {label:15} cosine {a['cosine']:.5f}  rel-L2 {a['rel_l2_percent']:.2f}%  [{a['verdict']}]")

    for key in ("profile_job", "inference_job"):
        if key in report:
            print(f"  {key.replace('_', ' ').title():15} {report[key]}")
    print(f"\n  Report: {json_path}\n{bar}")


if __name__ == "__main__":
    main()
