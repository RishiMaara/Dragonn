"""
Hexagon Bridge — QNN EP Session Profiler
=========================================
Step 3: Run the quantized model through ONNX Runtime with QNN EP enabled
and DON'T disable CPU fallback. Log which nodes land on QNN vs CPU.

This gives us the REAL "before" number — not our static analysis, but
actual ORT node placement data.

ORT's session profiling tells you exactly which EP handled each node.
We parse that into our coverage report format for dashboard consumption.

Usage:
    # On Snapdragon X ARM64 device only (QNN EP requires Hexagon NPU):
    python -m scripts.profile_qnn --input ./models/whisper-medium-int8/encoder_model.onnx

    # Dry run on x86 (CPU EP only, validates the profiling code works):
    python -m scripts.profile_qnn --input ./models/whisper-medium-int8/encoder_model.onnx --dry-run
"""

import argparse
import json
import logging
import sys
import time
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("hexagon-bridge.profiler")


def check_qnn_available() -> bool:
    """
    True only if QNN EP can EXECUTE on a Hexagon NPU on this machine.

    Not `"QNNExecutionProvider" in ort.get_available_providers()`: the
    onnxruntime-qnn 2.x plugin is absent from that list until registered, and on
    x64 it registers but can only compile, not execute. See scripts/qnn_ep.py.
    """
    try:
        from scripts.qnn_ep import npu_available
        return npu_available()
    except ImportError:
        return False


def profile_model_on_qnn(
    model_path: str | Path,
    qnn_available: bool = True,
    num_warmup_runs: int = 3,
    num_profile_runs: int = 10,
    enable_profiling: bool = True,
) -> dict:
    """
    Run a model through ONNX Runtime with QNN EP and CPU EP fallback,
    and capture per-node execution provider assignment via session profiling.

    Args:
        model_path:       Path to the quantized ONNX model
        qnn_available:    If False, run CPU-only (for baseline comparison)
        num_warmup_runs:  Number of warmup inferences before profiling
        num_profile_runs: Number of profiled inferences

    Returns:
        dict with:
          - node_placements: {node_name: "QNNExecutionProvider" | "CPUExecutionProvider"}
          - ep_distribution: {"QNN": count, "CPU": count}
          - latency_stats: {mean_ms, min_ms, max_ms, std_ms}
          - profile_data: raw profiling events (if enabled)
    """
    import onnxruntime as ort

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Profiling: {model_path.name}")
    available_providers = ort.get_available_providers()
    logger.info(f"Available EPs: {available_providers}")

    use_qnn = qnn_available and check_qnn_available()
    qnn_options = {
        "htp_performance_mode": "burst",
        "htp_graph_finalization_optimization_mode": "3",
        "enable_htp_fp16_precision": "1",
    }
    if use_qnn:
        logger.info("Using: QNN EP (HTP) + CPU EP for any nodes QNN rejects")
    else:
        logger.info("Using: CPU EP only (baseline mode)")

    # Create session options with profiling enabled
    sess_options = ort.SessionOptions()

    if enable_profiling:
        sess_options.enable_profiling = True
        # Profile file will be saved to temp directory
        profile_dir = tempfile.mkdtemp(prefix="hexbridge_profile_")
        sess_options.profile_file_prefix = str(
            Path(profile_dir) / "qnn_profile"
        )
        logger.info(f"Profile output: {profile_dir}")

    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Create session
    logger.info("Creating inference session...")
    start_time = time.time()

    try:
        if use_qnn:
            from scripts.qnn_ep import create_session
            session = create_session(model_path, qnn_options, sess_options)
        else:
            session = ort.InferenceSession(
                str(model_path), sess_options=sess_options, providers=["CPUExecutionProvider"]
            )
    except Exception as e:
        if use_qnn:
            # No silent CPU fallback here: a CPU session profiled under the QNN
            # label is exactly the misleading number this project exists to stop.
            logger.error(
                f"QNN EP session failed: {e}\n"
                "Usual causes:\n"
                "  1. x64 Python on ARM64 Windows (Prism) — QnnHtp.dll cannot load\n"
                "  2. onnxruntime-qnn not installed (pip install -r requirements-device.txt)\n"
                "  3. The HTP compiler rejected the graph — run the scanner, and\n"
                "     python -m scripts.aihub_validate to see which ops"
            )
        raise

    session_create_time = time.time() - start_time
    logger.info(f"Session created in {session_create_time:.2f}s")

    # Generate dummy input data matching model's input spec
    feeds = {}
    for inp in session.get_inputs():
        shape = []
        for dim in inp.shape:
            if isinstance(dim, int):
                shape.append(dim)
            elif isinstance(dim, str):
                # Dynamic dimension — use defaults
                shape.append(_default_dynamic_dim(inp.name, dim))
            else:
                shape.append(1)

        dtype = _onnx_type_to_numpy(inp.type)
        if dtype in (np.int32, np.int64):
            feeds[inp.name] = np.random.randint(0, 100, size=shape, dtype=dtype)
        else:
            feeds[inp.name] = np.random.randn(*shape).astype(dtype)

        logger.debug(f"  Input '{inp.name}': shape={shape}, dtype={dtype}")

    # Warmup runs
    logger.info(f"Running {num_warmup_runs} warmup inferences...")
    for _ in range(num_warmup_runs):
        session.run(None, feeds)

    # Profiled runs with latency measurement
    logger.info(f"Running {num_profile_runs} profiled inferences...")
    latencies = []
    for i in range(num_profile_runs):
        t0 = time.perf_counter()
        session.run(None, feeds)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # Convert to ms
        logger.debug(f"  Run {i+1}: {latencies[-1]:.2f} ms")

    # Collect profiling data
    profile_file = None
    profile_data = []
    node_placements = {}
    ep_distribution = Counter()

    if enable_profiling:
        try:
            profile_file = session.end_profiling()
            logger.info(f"Profile saved to: {profile_file}")

            with open(profile_file, "r") as f:
                profile_data = json.load(f)

            # Parse profiling events to determine EP assignment per node
            for event in profile_data:
                if not isinstance(event, dict):
                    continue

                name = event.get("name", "")
                cat = event.get("cat", "")
                args = event.get("args", {})

                # ORT profiling events have 'cat' = 'Node' for compute nodes
                # and 'args.provider' tells which EP ran it
                if cat == "Node" and "provider" in args:
                    node_name = name
                    provider = args["provider"]
                    node_placements[node_name] = provider
                    ep_distribution[provider] += 1

                # Also look for 'op_name' in args
                elif cat == "Node" and "op_name" in args:
                    node_name = args.get("op_name", name)
                    provider = args.get("provider", "unknown")
                    if provider != "unknown":
                        node_placements[node_name] = provider
                        ep_distribution[provider] += 1

        except Exception as e:
            logger.warning(f"Failed to parse profiling data: {e}")

    # Calculate latency stats
    latency_stats = {
        "mean_ms": round(np.mean(latencies), 2),
        "min_ms": round(np.min(latencies), 2),
        "max_ms": round(np.max(latencies), 2),
        "std_ms": round(np.std(latencies), 2),
        "median_ms": round(np.median(latencies), 2),
        "p95_ms": round(np.percentile(latencies, 95), 2),
        "num_runs": num_profile_runs,
        "all_latencies_ms": [round(l, 2) for l in latencies],
    }

    # Summarize EP distribution
    total_profiled_nodes = sum(ep_distribution.values())
    ep_summary = {}
    for ep, count in ep_distribution.most_common():
        ep_name = ep.replace("ExecutionProvider", "")
        ep_summary[ep_name] = {
            "count": count,
            "percent": round(count / total_profiled_nodes * 100, 1)
            if total_profiled_nodes > 0
            else 0,
        }

    result = {
        "model_path": str(model_path.resolve()),
        "model_name": model_path.stem,
        "session_creation_time_ms": round(session_create_time * 1000, 2),
        # What actually attached, not what was requested — they can differ silently.
        "providers_used": session.get_providers(),
        "node_placements": node_placements,
        "ep_distribution": dict(ep_distribution),
        "ep_summary": ep_summary,
        "total_profiled_nodes": total_profiled_nodes,
        "latency_stats": latency_stats,
        "profile_file": profile_file,
    }

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"PROFILING RESULTS: {model_path.name}")
    logger.info(f"{'='*60}")
    logger.info(f"  Session creation: {session_create_time*1000:.0f} ms")
    logger.info(f"  Mean latency:     {latency_stats['mean_ms']:.2f} ms")
    logger.info(f"  P95 latency:      {latency_stats['p95_ms']:.2f} ms")
    if ep_summary:
        logger.info(f"  EP Distribution:")
        for ep_name, stats in ep_summary.items():
            logger.info(f"    {ep_name}: {stats['count']} nodes ({stats['percent']}%)")
    else:
        logger.info("  EP Distribution: No profiling data available")
    logger.info(f"{'='*60}")

    return result


def run_cpu_baseline(
    model_path: str | Path,
    num_warmup_runs: int = 3,
    num_profile_runs: int = 10,
) -> dict:
    """
    Run the model on CPU EP only — this is our "before" number.
    Same model, CPU only, no QNN at all.
    """
    logger.info("Running CPU-only baseline...")
    return profile_model_on_qnn(
        model_path=model_path,
        qnn_available=False,
        num_warmup_runs=num_warmup_runs,
        num_profile_runs=num_profile_runs,
        enable_profiling=False,
    )


def compare_results(cpu_result: dict, qnn_result: dict) -> dict:
    """
    Generate a before/after comparison between CPU-only and QNN EP runs.
    This is the "receipt" — the honest number judges see.
    """
    cpu_latency = cpu_result["latency_stats"]["mean_ms"]
    qnn_latency = qnn_result["latency_stats"]["mean_ms"]

    speedup = cpu_latency / qnn_latency if qnn_latency > 0 else 0
    improvement_pct = (
        (cpu_latency - qnn_latency) / cpu_latency * 100 if cpu_latency > 0 else 0
    )

    comparison = {
        "cpu_baseline": {
            "mean_latency_ms": cpu_latency,
            "p95_latency_ms": cpu_result["latency_stats"]["p95_ms"],
        },
        "qnn_optimized": {
            "mean_latency_ms": qnn_latency,
            "p95_latency_ms": qnn_result["latency_stats"]["p95_ms"],
            "ep_distribution": qnn_result.get("ep_summary", {}),
        },
        "improvement": {
            "speedup_factor": round(speedup, 2),
            "latency_reduction_percent": round(improvement_pct, 1),
        },
    }

    print("\n" + "=" * 60)
    print("  BEFORE / AFTER COMPARISON")
    print("=" * 60)
    print(f"  CPU-only baseline:  {cpu_latency:.2f} ms (mean)")
    print(f"  QNN EP optimized:   {qnn_latency:.2f} ms (mean)")
    print(f"  Speedup:            {speedup:.2f}x")
    print(f"  Improvement:        {improvement_pct:.1f}%")
    if qnn_result.get("ep_summary"):
        print(f"  NPU utilization:    ", end="")
        for ep, stats in qnn_result["ep_summary"].items():
            print(f"{ep}: {stats['percent']}%  ", end="")
        print()
    print("=" * 60)

    return comparison


def _default_dynamic_dim(input_name: str, dim_name: str) -> int:
    """Provide sensible defaults for dynamic dimensions."""
    defaults = {
        "batch_size": 1,
        "batch": 1,
        "sequence_length": 1,
        "seq_len": 1,
        "encoder_sequence_length": 1500,
        "decoder_sequence_length": 1,
        "num_mel_bins": 80,
        "feature_size": 80,
    }

    dim_lower = dim_name.lower()
    for key, val in defaults.items():
        if key in dim_lower:
            return val

    if "past" in dim_lower or "cache" in dim_lower:
        return 0

    return 1


def _onnx_type_to_numpy(ort_type: str):
    """Convert ORT type string to numpy dtype."""
    type_map = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
        "tensor(double)": np.float64,
    }
    return type_map.get(ort_type, np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Hexagon Bridge — QNN EP Session Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile on Snapdragon device (QNN EP available)
  python -m scripts.profile_qnn --input ./models/whisper-medium-int8/encoder_model.onnx

  # Dry run on x86 (CPU EP only, validates code)
  python -m scripts.profile_qnn --input ./models/whisper-medium-int8/encoder_model.onnx --dry-run

  # Full before/after comparison
  python -m scripts.profile_qnn --input ./models/whisper-medium-int8/encoder_model.onnx --compare

  # Save results as JSON
  python -m scripts.profile_qnn --input ./models/whisper-medium-int8/encoder_model.onnx --json ./results/profile.json
        """,
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to quantized ONNX model"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run CPU-only (when QNN EP not available)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both CPU and QNN EP for before/after comparison",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup runs"
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of profiled runs"
    )
    parser.add_argument(
        "--json", type=str, default=None, help="Save results to JSON file"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    qnn_ok = check_qnn_available() and not args.dry_run

    if args.compare:
        # Before/after comparison
        cpu_result = run_cpu_baseline(
            args.input, args.warmup, args.runs
        )
        qnn_result = profile_model_on_qnn(
            args.input, qnn_ok, args.warmup, args.runs
        )
        comparison = compare_results(cpu_result, qnn_result)

        if args.json:
            output_path = Path(args.json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(
                    {
                        "cpu_baseline": cpu_result,
                        "qnn_optimized": qnn_result,
                        "comparison": comparison,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            print(f"\nResults saved to: {output_path}")
    else:
        result = profile_model_on_qnn(
            args.input, qnn_ok, args.warmup, args.runs
        )

        if args.json:
            output_path = Path(args.json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
