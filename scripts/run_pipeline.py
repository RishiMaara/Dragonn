"""
Hexagon Bridge — Full Pipeline Runner
======================================
One-command orchestrator that runs the entire pipeline:
  1. Export Whisper-Medium from HuggingFace → ONNX
  2. Quantize to INT8 QDQ format
  3. Scan QNN operator coverage
  4. Profile on QNN EP (or CPU baseline)
  5. Generate combined report

Usage:
    # Full pipeline (on x86 for export/quantize, then transfer to ARM64 for profiling)
    python -m scripts.run_pipeline --model openai/whisper-medium

    # Skip export if already have ONNX files
    python -m scripts.run_pipeline --skip-export --onnx-dir ./models/whisper-medium-onnx

    # Quick test with whisper-small
    python -m scripts.run_pipeline --model openai/whisper-small --quick
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger("hexagon-bridge.pipeline")


def run_full_pipeline(
    model_id: str = "openai/whisper-medium",
    output_base: str = "./models",
    skip_export: bool = False,
    skip_quantize: bool = False,
    onnx_dir: str | None = None,
    quantized_dir: str | None = None,
    quick: bool = False,
) -> dict:
    """
    Run the complete Hexagon Bridge pipeline.

    Returns:
        dict with results from each stage
    """
    results = {}
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    model_short = model_id.split("/")[-1]

    # ── Stage 1: Export HF → ONNX ──
    if not skip_export:
        logger.info("\n" + "=" * 70)
        logger.info("  STAGE 1: HuggingFace → ONNX Export")
        logger.info("=" * 70)

        from converter.hf_to_onnx import export_whisper_to_onnx

        onnx_output = onnx_dir or str(output_base / f"{model_short}-onnx")

        export_result = export_whisper_to_onnx(
            model_id=model_id,
            output_dir=onnx_output,
            opset=17,
        )
        results["export"] = export_result
        onnx_dir = onnx_output
    else:
        logger.info("Skipping export (using existing ONNX files)")
        if not onnx_dir:
            onnx_dir = str(output_base / f"{model_short}-onnx")

    # ── Stage 2: INT8 QDQ Quantization ──
    if not skip_quantize:
        logger.info("\n" + "=" * 70)
        logger.info("  STAGE 2: INT8 QDQ Quantization")
        logger.info("=" * 70)

        from converter.quantize import quantize_whisper_pipeline

        quant_output = quantized_dir or str(output_base / f"{model_short}-int8")

        # Real speech sets far better activation ranges than synthetic audio: on
        # 57 held-out LibriSpeech clips it took the quantized encoder from +0.63
        # to +0.00 WER vs the FP32 reference. Fetch it once with
        # `python -m scripts.fetch_speech`.
        calib_dir = Path("data/speech/calib")
        if not calib_dir.is_dir():
            logger.warning(
                "No real calibration speech at data/speech/calib — falling back to "
                "synthetic audio (measured +0.63 WER). Run: python -m scripts.fetch_speech"
            )
        quantize_result = quantize_whisper_pipeline(
            input_dir=onnx_dir,
            output_dir=quant_output,
            weight_type="UINT8",
            calibrate=not quick,
            num_calibration_samples=16 if calib_dir.is_dir() else (20 if quick else 50),
            audio_dir=calib_dir if calib_dir.is_dir() else None,
        )
        results["quantize"] = quantize_result
        quantized_dir = quant_output
    else:
        logger.info("Skipping quantization (using existing quantized files)")
        if not quantized_dir:
            quantized_dir = str(output_base / f"{model_short}-int8")

    # ── Stage 3: QNN Op Coverage Scan ──
    logger.info("\n" + "=" * 70)
    logger.info("  STAGE 3: QNN Operator Coverage Scan")
    logger.info("=" * 70)

    from scanner.graph_analyzer import analyze_directory
    from scanner.report import print_coverage_report, save_json_report, generate_summary_text

    scan_target = quantized_dir or onnx_dir
    coverage_reports = analyze_directory(scan_target)

    # Static analysis can't see per-node encoding rejections; the real HTP
    # compiler can, and runs on x64 when onnxruntime-qnn is installed.
    from scripts.qnn_ep import compile_only_available
    if compile_only_available():
        from scanner.htp_compile import compile_check
        for filename, report in coverage_reports.items():
            logger.info(f"Compiling {filename} with the local HTP compiler...")
            report.htp_compile = compile_check(report.model_path)
    else:
        logger.info("Local HTP compile check skipped (pip install onnxruntime-qnn to enable)")

    for filename, report in coverage_reports.items():
        print_coverage_report(report)

    # Save JSON report for dashboard consumption
    reports_dir = output_base / "reports"
    reports_dir.mkdir(exist_ok=True)
    json_report_path = reports_dir / "coverage_report.json"
    save_json_report(coverage_reports, json_report_path)

    # Generate pitch summary
    summary = generate_summary_text(coverage_reports)
    summary_path = reports_dir / "pitch_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    results["coverage"] = {
        filename: report.to_dict() for filename, report in coverage_reports.items()
    }
    results["coverage_summary"] = summary

    # ── Stage 4: Profile (if QNN available) ──
    logger.info("\n" + "=" * 70)
    logger.info("  STAGE 4: Performance Profiling")
    logger.info("=" * 70)

    from scripts.profile_qnn import (
        check_qnn_available,
        profile_model_on_qnn,
        run_cpu_baseline,
        compare_results,
    )

    qnn_available = check_qnn_available()

    if qnn_available:
        logger.info("QNN EP detected — running before/after comparison")

        # Find the encoder model (typically the largest / most important)
        encoder_path = Path(scan_target) / "encoder_model.onnx"
        if not encoder_path.exists():
            # Fall back to first .onnx file
            onnx_files = list(Path(scan_target).glob("*.onnx"))
            encoder_path = onnx_files[0] if onnx_files else None

        if encoder_path and encoder_path.exists():
            cpu_result = run_cpu_baseline(str(encoder_path), num_profile_runs=5)
            qnn_result = profile_model_on_qnn(
                str(encoder_path), True, num_profile_runs=5
            )
            comparison = compare_results(cpu_result, qnn_result)

            results["profiling"] = {
                "cpu_baseline": cpu_result,
                "qnn_optimized": qnn_result,
                "comparison": comparison,
            }

            # Save profiling results
            profile_path = reports_dir / "profiling_results.json"
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(results["profiling"], f, indent=2, default=str)
    else:
        logger.info(
            "QNN EP not available on this machine.\n"
            "  → CPU-only baseline will be captured.\n"
            "  → Transfer quantized models to Snapdragon X device for QNN profiling.\n"
            f"  → Quantized models are in: {scan_target}"
        )

        # Still run CPU baseline for reference
        encoder_path = Path(scan_target) / "encoder_model.onnx"
        if encoder_path.exists():
            cpu_result = run_cpu_baseline(str(encoder_path), num_profile_runs=5)
            results["profiling"] = {"cpu_baseline": cpu_result}

    # ── Save combined results ──
    combined_path = reports_dir / "pipeline_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info(f"  PIPELINE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Reports saved to: {reports_dir}")
    logger.info(f"  Coverage report:  {json_report_path}")
    logger.info(f"  Pitch summary:    {summary_path}")
    logger.info(f"  Combined results: {combined_path}")
    if not qnn_available:
        logger.info(f"\n  ⚠️  Transfer {scan_target}/ to Snapdragon X device")
        logger.info(f"     then run: python -m scripts.profile_qnn --input <path> --compare")
    logger.info(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Hexagon Bridge — Full Pipeline Runner",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-medium",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models",
        help="Base output directory",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX export (use existing files)",
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip quantization (use existing files)",
    )
    parser.add_argument(
        "--onnx-dir",
        type=str,
        default=None,
        help="Path to existing ONNX files (with --skip-export)",
    )
    parser.add_argument(
        "--quantized-dir",
        type=str,
        default=None,
        help="Path to existing quantized files (with --skip-quantize)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (fewer calibration samples)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    import sys
    import io

    # Force UTF-8 encoding for standard output to prevent PyTorch's emojis from crashing Windows cp1252 consoles
    if isinstance(sys.stdout.buffer, io.BufferedWriter):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_full_pipeline(
        model_id=args.model,
        output_base=args.output,
        skip_export=args.skip_export,
        skip_quantize=args.skip_quantize,
        onnx_dir=args.onnx_dir,
        quantized_dir=args.quantized_dir,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
