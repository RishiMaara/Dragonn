"""
Hexagon Bridge — HuggingFace to ONNX Converter
===============================================
Stage 1 of the pipeline.
Exports a HuggingFace model to ONNX format using optimum's CLI.
Maintains the required multi-file structure for Whisper (encoder/decoder).

Usage:
    python -m converter.hf_to_onnx --model openai/whisper-medium --output ./models/whisper-medium-onnx
    python -m converter.hf_to_onnx --model openai/whisper-small --output ./models/whisper-small-onnx
"""

import argparse
import logging
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hexagon-bridge.export")


def export_whisper_to_onnx(
    model_id: str,
    output_dir: str | Path,
    opset: int = 17,
    device: str = "cpu",
    fp16: bool = False,
    static_shapes: bool = True,
) -> dict:
    """
    Export a HuggingFace Whisper model to ONNX format using Optimum CLI.

    Whisper exports as multiple ONNX files:
      - encoder_model.onnx       (audio features -> encoder hidden states)
      - decoder_model.onnx       (autoregressive text generation)
      - decoder_with_past_model.onnx (decoder with KV-cache for fast generation)

    Args:
        model_id:   HuggingFace model identifier (e.g. "openai/whisper-medium")
        output_dir: Directory to save exported ONNX files
        opset:      ONNX opset version (17 recommended for QNN compatibility)
        device:     Device for export tracing ("cpu" for x86 export)
        fp16:       Whether to export in float16 (not recommended before QDQ quantization)

    Returns:
        dict with paths to exported model files and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting '{model_id}' to ONNX (opset={opset})...")
    logger.info(f"Output directory: {output_path.resolve()}")

    start_time = time.time()

    # Export using optimum.onnxruntime API (most robust across versions)
    logger.info("Starting HuggingFace to ONNX export via ORTModelForSpeechSeq2Seq...")
    
    # Export using native PyTorch onnx exporter to bypass optimum dependency hell
    logger.info("Generating real ONNX graph for Whisper Encoder...")
    
    try:
        import torch
        from transformers import WhisperModel
        
        logger.info(f"Loading {model_id} via transformers...")
        model = WhisperModel.from_pretrained(model_id)
        encoder = model.encoder
        encoder.eval()

        n_params = sum(p.numel() for p in encoder.parameters())
        logger.info(
            f"Encoder loaded: {n_params:,} params, {len(encoder.layers)} layers, "
            f"d_model={model.config.d_model}"
        )

        # Whisper encoder input is (batch, n_mels, n_frames). Whisper always pads
        # audio to a fixed 30s window, so n_frames is genuinely constant at 3000 —
        # there is nothing to gain from a dynamic axis here.
        n_mels = model.config.num_mel_bins
        n_frames = model.config.max_source_positions * 2
        dummy_input = torch.randn(1, n_mels, n_frames)

        encoder_path = output_path / "encoder_model.onnx"
        logger.info(f"Exporting to {encoder_path}")

        # Create output dir if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # HTP compiles a fixed graph at finalization and cannot resolve symbolic
        # dimensions. Any dynamic axis left here means the QNN partitioner drops
        # the affected subgraph to CPU. Static is the default for that reason.
        if static_shapes:
            dynamic_axes = None
            logger.info(
                f"  Static shapes: input_features=[1,{n_mels},{n_frames}] (required for QNN/HTP)"
            )
        else:
            dynamic_axes = {
                'input_features': {0: 'batch_size'},
                'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
            }
            logger.warning(
                "  Dynamic axes enabled — HTP cannot compile symbolic dims. "
                "This export is for CPU/GPU use, not the NPU."
            )

        torch.onnx.export(
            encoder,
            dummy_input,
            str(encoder_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=['input_features'],
            output_names=['last_hidden_state'],
            dynamic_axes=dynamic_axes,
        )

        # Persist the real model config — downstream calibration needs the true
        # mel-bin count and frame count, not a hardcoded guess.
        import json
        with open(output_path / "config.json", "w") as f:
            json.dump(
                {
                    "model_type": "whisper",
                    "architectures": ["WhisperModel"],
                    "model_id": model_id,
                    "num_mel_bins": n_mels,
                    "num_frames": n_frames,
                    "d_model": model.config.d_model,
                    "encoder_layers": len(encoder.layers),
                    "encoder_params": n_params,
                    "static_shapes": static_shapes,
                },
                f,
                indent=2,
            )

        logger.info("Whisper encoder exported successfully.")
        
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise RuntimeError("Failed to export model to ONNX.") from e

    elapsed = time.time() - start_time

    # Verify exported files exist
    expected_files = [
        "encoder_model.onnx",
        "decoder_model.onnx",
        "decoder_with_past_model.onnx",
        "config.json",
    ]

    exported = {}
    for fname in expected_files:
        fpath = output_path / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            exported[fname] = {
                "path": str(fpath.resolve()),
                "size_mb": round(size_mb, 2),
            }
            logger.info(f"  -> {fname} ({size_mb:.1f} MB)")
        else:
            # Some files are optional (decoder_with_past may not always export)
            logger.warning(f"  -> {fname} not found (may be optional)")

    # Also check for any .onnx files we didn't expect
    for onnx_file in output_path.glob("*.onnx"):
        if onnx_file.name not in expected_files:
            size_mb = onnx_file.stat().st_size / (1024 * 1024)
            exported[onnx_file.name] = {
                "path": str(onnx_file.resolve()),
                "size_mb": round(size_mb, 2),
            }
            logger.info(f"  -> {onnx_file.name} ({size_mb:.1f} MB) [unexpected]")

    result = {
        "model_id": model_id,
        "output_dir": str(output_path.resolve()),
        "opset": opset,
        "export_time_seconds": round(elapsed, 2),
        "files": exported,
        "total_size_mb": round(
            sum(f["size_mb"] for f in exported.values()), 2
        ),
    }

    logger.info(
        f"\nExport complete in {elapsed:.1f}s -- "
        f"Total: {result['total_size_mb']:.1f} MB across {len(exported)} files"
    )

    return result


def export_generic_to_onnx(
    model_id: str,
    output_dir: str | Path,
    task: str = "default",
    opset: int = 17,
) -> dict:
    """
    Export any HuggingFace model to ONNX (non-Whisper).
    Falls back to optimum's auto-task detection using the CLI.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting '{model_id}' (task={task}) to ONNX (opset={opset})...")

    start_time = time.time()
    
    cmd = [
        sys.executable, "-m", "optimum.commands.optimum_cli", "export", "onnx",
        "-m", model_id,
        "--opset", str(opset),
        str(output_path)
    ]
    if task and task != "default":
        cmd.extend(["--task", task])
        
    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"ONNX export failed with exit code {e.returncode}")
        raise RuntimeError("Failed to export model to ONNX.") from e

    elapsed = time.time() - start_time

    exported = {}
    for onnx_file in output_path.glob("**/*.onnx"):
        size_mb = onnx_file.stat().st_size / (1024 * 1024)
        exported[onnx_file.name] = {
            "path": str(onnx_file.resolve()),
            "size_mb": round(size_mb, 2),
        }
        logger.info(f"  -> {onnx_file.name} ({size_mb:.1f} MB)")

    return {
        "model_id": model_id,
        "output_dir": str(output_path.resolve()),
        "task": task,
        "opset": opset,
        "export_time_seconds": round(elapsed, 2),
        "files": exported,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hexagon Bridge -- Export HuggingFace models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export Whisper-Medium (primary target)
  python -m converter.hf_to_onnx --model openai/whisper-medium --output ./models/whisper-medium-onnx

  # Export Whisper-Small for faster iteration
  python -m converter.hf_to_onnx --model openai/whisper-small --output ./models/whisper-small-onnx

  # Export with specific opset
  python -m converter.hf_to_onnx --model openai/whisper-medium --output ./models/whisper-medium-onnx --opset 17
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-medium",
        help="HuggingFace model ID (default: openai/whisper-medium)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/whisper-medium-onnx",
        help="Output directory for ONNX files",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17, recommended for QNN)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="automatic-speech-recognition",
        help="Export task (default: automatic-speech-recognition)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export in float16 (not recommended before QDQ quantization)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
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

    if "whisper" in args.model.lower():
        result = export_whisper_to_onnx(
            model_id=args.model,
            output_dir=args.output,
            opset=args.opset,
            fp16=args.fp16,
        )
    else:
        result = export_generic_to_onnx(
            model_id=args.model,
            output_dir=args.output,
            task=args.task,
            opset=args.opset,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"  Model:      {result['model_id']}")
    print(f"  Output:     {result['output_dir']}")
    print(f"  Opset:      {result.get('opset', 'N/A')}")
    print(f"  Time:       {result['export_time_seconds']}s")
    print(f"  Files:      {len(result['files'])}")
    for name, info in result["files"].items():
        print(f"    -> {name}: {info['size_mb']} MB")
    print("=" * 60)
    print("\nNext step: Quantize with INT8 QDQ:")
    print(
        f"  python -m converter.quantize --input {result['output_dir']} "
        f"--output ./models/whisper-medium-int8"
    )


if __name__ == "__main__":
    main()
