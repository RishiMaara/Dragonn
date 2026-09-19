"""
Hexagon Bridge — INT8 QDQ Quantization
=======================================
Step 2 of the pipeline: Quantize ONNX models to STATIC QDQ
(QuantizeLinear/DequantizeLinear) format — the only format QNN EP can consume.

Runs on any host (the ORT quantization tools are pure Python); typically the
x64 prep host, since calibration wants PyTorch and real input data.

QDQ quantization inserts QuantizeLinear/DequantizeLinear node pairs around
compute-heavy ops (MatMul, Conv, etc.), forming the "QDQ node units" QNN EP
fuses into fixed-point operations on the Hexagon NPU.

WHAT NOT TO DO
--------------
Dynamic quantization (quantize_dynamic) emits DynamicQuantizeLinear,
MatMulInteger and ConvInteger. QNN EP has no builders for any of them. The
resulting model loads cleanly on a Snapdragon device, returns correct outputs,
and runs 100% on CPU — with no error, warning or diagnostic anywhere. That is
the single most common way a model silently never reaches the NPU.

Activations must be UNSIGNED (uint8/uint16). Symmetric int8 activations are the
other common wrong answer.

Usage:
    python -m converter.quantize --input ./models/whisper-medium-onnx --output ./models/whisper-medium-int8
    python -m converter.quantize --input ./models/whisper-medium-onnx --output ./models/whisper-medium-int8 --calibrate
"""

import argparse
import logging
import sys
import time
import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("hexagon-bridge.quantize")


class WhisperCalibrationDataReader:
    """
    Provides calibration data for static quantization of Whisper models.

    Static QDQ quantization needs representative input data to fix the
    quantization range (min/max) of every activation tensor. Those ranges are
    baked into the model permanently, so bad calibration data means a
    structurally perfect graph that computes garbage.

    Features are produced by running audio through Whisper's own
    WhisperFeatureExtractor — real audio when `audio_dir` is supplied, otherwise
    synthetic speech-like audio. What matters is that features come out of the
    real extractor, so they carry the correct dynamic range and floor.

    Pass real audio for anything you intend to ship. Synthetic audio gets the
    distribution roughly right; it does not get the content right.
    """

    def __init__(
        self,
        model_path: str,
        num_samples: int = 100,
        use_real_data: bool = False,
        model_id: str = "openai/whisper-tiny",
        audio_dir: str | Path | None = None,
    ):
        import onnx

        self.num_samples = num_samples
        self.sample_index = 0
        self.model_path = model_path
        self.model_id = model_id

        # Calibration decides every activation scale in the model. Feeding it
        # np.random.randn is the single fastest way to destroy accuracy while
        # keeping the graph structurally perfect: real Whisper log-mel features
        # sit in roughly [-0.6, 1.4] with std ~0.3, while standard normal noise
        # spans [-5, 5] with std 1.0. Calibrating on the latter picks ranges ~5x
        # too wide, and every tensor loses most of its effective bit depth.
        #
        # So: synthesize AUDIO, then push it through the real feature extractor.
        # Even synthetic speech gives correctly-shaped features; random numbers
        # in feature space do not.
        self._feature_extractor = None
        self._audio_files: list[Path] = []

        if audio_dir:
            audio_path = Path(audio_dir)
            if audio_path.is_dir():
                for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
                    self._audio_files.extend(sorted(audio_path.glob(ext)))
                logger.info(f"  Calibration audio: {len(self._audio_files)} files from {audio_path}")

        try:
            from transformers import WhisperFeatureExtractor
            self._feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
            self._sampling_rate = self._feature_extractor.sampling_rate
            logger.info(
                f"  Calibration via WhisperFeatureExtractor ({model_id}) — "
                f"{'real audio' if self._audio_files else 'synthetic audio'}"
            )
        except Exception as e:
            logger.warning(
                f"  WhisperFeatureExtractor unavailable ({e}). Falling back to raw "
                "random features — EXPECT SEVERE ACCURACY LOSS. Install transformers "
                "or pass --audio-dir with real samples."
            )

        # Load model to inspect input shapes
        model = onnx.load(model_path)
        self.input_specs = {}
        for inp in model.graph.input:
            name = inp.name
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    # Dynamic dimension — use sensible defaults for Whisper
                    shape.append(self._default_dim(name, len(shape)))
            dtype = self._onnx_dtype_to_numpy(inp.type.tensor_type.elem_type)
            self.input_specs[name] = {"shape": shape, "dtype": dtype}

        logger.info(f"Calibration data reader initialized for: {model_path}")
        for name, spec in self.input_specs.items():
            logger.info(f"  Input '{name}': shape={spec['shape']}, dtype={spec['dtype']}")

    def _default_dim(self, input_name: str, dim_idx: int) -> int:
        """Sensible default dimensions for Whisper's dynamic axes."""
        # Whisper encoder: (batch, n_mels=80, n_frames=3000)
        # Whisper decoder: (batch, seq_len) — use 1 for calibration
        defaults = {
            "input_features": [1, 80, 3000],  # encoder input
            "decoder_input_ids": [1, 1],       # decoder input (single token)
            "encoder_hidden_states": [1, 1500, 1024],  # medium: 1024 hidden
        }

        for key, dims in defaults.items():
            if key in input_name and dim_idx < len(dims):
                return dims[dim_idx]

        # Fallback
        return 1 if dim_idx == 0 else 64

    def _onnx_dtype_to_numpy(self, onnx_dtype: int):
        """Map ONNX TensorProto data types to numpy dtypes."""
        type_map = {
            1: np.float32,
            2: np.uint8,
            3: np.int8,
            5: np.int16,
            6: np.int32,
            7: np.int64,
            10: np.float16,
            11: np.float64,
        }
        return type_map.get(onnx_dtype, np.float32)

    def get_next(self) -> Optional[dict]:
        """Return the next calibration sample, or None when exhausted."""
        if self.sample_index >= self.num_samples:
            return None

        idx = self.sample_index
        self.sample_index += 1

        feed = {}
        for name, spec in self.input_specs.items():
            if spec["dtype"] in (np.int32, np.int64):
                # For token IDs, use random valid token indices
                feed[name] = np.random.randint(
                    0, 51865, size=spec["shape"], dtype=spec["dtype"]
                )  # 51865 = Whisper vocab size
            elif "input_features" in name and self._feature_extractor is not None:
                feed[name] = self._make_log_mel(spec, idx)
            else:
                feed[name] = np.random.randn(*spec["shape"]).astype(spec["dtype"])

        return feed

    def _make_log_mel(self, spec: dict, idx: int) -> np.ndarray:
        """
        Produce a genuine log-mel feature tensor by running audio through
        Whisper's own feature extractor — the only way to get the right
        dynamic range, floor and normalization for calibration.
        """
        batch = spec["shape"][0] if spec["shape"] else 1
        frames = []

        for b in range(max(1, batch)):
            audio = self._next_audio(idx * max(1, batch) + b)
            out = self._feature_extractor(
                audio,
                sampling_rate=self._sampling_rate,
                return_tensors="np",
            )["input_features"]
            frames.append(out[0])

        feat = np.stack(frames).astype(spec["dtype"])

        # Match the graph's expected frame count if the extractor's padding differs.
        if len(spec["shape"]) == 3 and feat.shape[2] != spec["shape"][2]:
            target = spec["shape"][2]
            if feat.shape[2] > target:
                feat = feat[:, :, :target]
            else:
                feat = np.pad(feat, ((0, 0), (0, 0), (0, target - feat.shape[2])))

        return feat

    def _next_audio(self, idx: int) -> np.ndarray:
        """Real audio sample if available, else varied speech-like synthesis."""
        if self._audio_files:
            import soundfile as sf
            path = self._audio_files[idx % len(self._audio_files)]
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != self._sampling_rate:
                import librosa
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=self._sampling_rate
                )
            return audio

        # Synthetic voiced speech: harmonic stack on a varying f0, shaped by a
        # syllable-rate envelope, plus breath noise and silence gaps. Crude, but
        # it lands in the right region of feature space — which is all
        # calibration needs.
        rng = np.random.default_rng(idx)
        sr = self._sampling_rate
        duration = 30.0  # Whisper always pads/truncates to a 30s window
        t = np.arange(int(duration * sr)) / sr

        f0 = rng.uniform(85, 255)  # adult speaking range
        audio = np.zeros_like(t)
        for harmonic in range(1, 9):
            audio += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)

        syllable_rate = rng.uniform(2.0, 6.0)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t + rng.uniform(0, 6.28))
        audio *= envelope
        audio += rng.normal(0, rng.uniform(0.005, 0.05), size=t.shape)

        # Silence gaps — real utterances are not wall-to-wall energy, and the
        # quiet floor matters for the low end of the activation range.
        for _ in range(rng.integers(2, 6)):
            start = rng.integers(0, len(t) - sr)
            audio[start:start + rng.integers(sr // 4, sr * 2)] *= 0.02

        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * rng.uniform(0.3, 0.95)

        return audio.astype(np.float32)

    def rewind(self):
        """Reset the reader to the beginning."""
        self.sample_index = 0


def _model_size_mb(model_path: Path) -> float:
    """
    Total on-disk size of a model, including any external-data sidecar.

    Weights above a size threshold live in a separate .onnx.data file rather than
    inside the .onnx protobuf. Measuring only the .onnx yields nonsense — a
    whisper-tiny encoder reads as 0.03 MB and its 4x INT8 compression looks like
    a 270x expansion.
    """
    total = model_path.stat().st_size

    try:
        import onnx
        model = onnx.load(str(model_path), load_external_data=False)
        locations = {
            entry.value
            for init in model.graph.initializer
            for entry in init.external_data
            if entry.key == "location"
        }
        for loc in locations:
            sidecar = model_path.parent / loc
            if sidecar.exists():
                total += sidecar.stat().st_size
    except Exception as e:
        logger.debug(f"Could not resolve external data for {model_path.name}: {e}")

    return total / (1024 * 1024)


def quantize_onnx_model(
    input_model_path: str | Path,
    output_model_path: str | Path,
    quant_format: str = "QDQ",
    weight_type: str = "UINT8",
    calibrate: bool = True,
    num_calibration_samples: int = 100,
    per_channel: bool = False,
    optimize_model: bool = True,
    model_id: str = "openai/whisper-tiny",
    audio_dir: str | Path | None = None,
    activation_type: str = "UINT16",
) -> dict:
    """
    Quantize an ONNX model to INT8 using QDQ format for QNN EP compatibility.

    QDQ format places QuantizeLinear/DequantizeLinear pairs around operations,
    which QNN EP can fuse into efficient integer operations on the Hexagon NPU.

    Args:
        input_model_path:        Path to the FP32 ONNX model
        output_model_path:       Path for the quantized output model
        quant_format:            "QDQ" (for QNN) or "QOperator" (for CPU)
        weight_type:             "INT8" or "UINT8"
        calibrate:               If True, use calibration data for activation quantization
        num_calibration_samples: Number of calibration samples to generate
        per_channel:             Per-channel quantization (more accurate, QNN supports it)
        optimize_model:          Whether to apply graph optimizations before quantization

    Returns:
        dict with quantization results and metadata
    """
    try:
        from onnxruntime.quantization import (
            QuantFormat,
            QuantType,
            quantize,
            quantize_static,
            quantize_dynamic,
            CalibrationMethod,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:
        logger.error(
            "onnxruntime quantization tools not found. "
            "Run: pip install onnxruntime"
        )
        sys.exit(1)

    # ORT ships QNN-specific quantization helpers that pick the encodings the
    # Hexagon NPU actually wants (uint8/uint16 activations, int8 weights, and
    # the op-specific overrides QNN needs). Strongly preferred over hand-rolling
    # the config — hand-rolled configs are how models end up silently on CPU.
    try:
        from onnxruntime.quantization.execution_providers.qnn import (
            get_qnn_qdq_config,
            qnn_preprocess_model,
        )
        HAS_QNN_HELPERS = True
    except ImportError:
        HAS_QNN_HELPERS = False
        logger.warning(
            "onnxruntime.quantization.execution_providers.qnn not available "
            "(needs onnxruntime >= 1.18). Falling back to a manually-specified "
            "QNN-compatible config — verify the result with the scanner."
        )

    input_path = Path(input_model_path)
    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input model not found: {input_path}")

    # Determine quantization types
    qformat = QuantFormat.QDQ if quant_format == "QDQ" else QuantFormat.QOperator
    # Weights default to UNSIGNED uint8 (ORT's QNN default). Signed int8 is fine
    # for Conv/MatMul, but the HTP LayerNorm op rejects a signed int8 gamma under
    # 16-bit activations. Measured on whisper-tiny: int8 weights left all 9
    # LayerNorms off-NPU (9 NPU partitions, and a hard "Failed to finalize QNN
    # graph" on a real Snapdragon X Elite, QNN 2.45); uint8 weights compile to a
    # single NPU partition at cosine 0.992 vs FP32.
    #
    # per_channel defaults to False for the same reason ORT's QNN config does:
    # it only targets Conv/ConvTranspose, and on this model it just produced
    # "Axis 1 is out-of-range" warnings on LayerNorm weights.
    wtype = QuantType.QInt8 if weight_type == "INT8" else QuantType.QUInt8

    # Activation precision is the single biggest accuracy lever on this pipeline.
    #
    # Measured on the whisper-tiny encoder (cosine similarity vs FP32, held-out
    # input, identical calibration set):
    #     uint8  activations -> 0.553   unusable
    #     uint16 activations -> 0.995   usable
    #
    # Transformer activations after LayerNorm span a wide range (this encoder
    # outputs roughly [-18, 18]); 256 levels cannot represent that without
    # destroying the signal. HTP supports 16-bit activations with 8-bit weights
    # ("a16w8"), which is why that is the default here. uint8 is offered for
    # size/speed comparisons, not for deployment.
    atype = QuantType.QUInt16 if activation_type.upper() == "UINT16" else QuantType.QUInt8

    input_size_mb = _model_size_mb(input_path)
    logger.info(f"Quantizing: {input_path.name} ({input_size_mb:.1f} MB)")
    logger.info(
        f"  Format: {quant_format}, Weight: {weight_type}, "
        f"Activation: {activation_type.upper()}, PerChannel: {per_channel}"
    )
    if activation_type.upper() == "UINT8":
        logger.warning(
            "  uint8 activations measured cosine 0.55 vs FP32 on this encoder. "
            "Use UINT16 for transformers unless you are benchmarking size."
        )

    start_time = time.time()

    # Step 1: Pre-process. Intermediates go to a scratch dir so they (and any
    # external-data sidecars) can never leak into the output directory.
    work_dir = Path(tempfile.mkdtemp(prefix="hexbridge_quant_"))
    preprocessed_path = work_dir / f"{input_path.stem}_pre.onnx"

    logger.info("  Step 1/3: Pre-processing...")
    if HAS_QNN_HELPERS:
        # qnn_preprocess_model REPLACES quant_pre_process for QNN rather than
        # following it: it runs its own shape inference and optimization, plus
        # the fusions HTP needs (Erf chain -> Gelu, ReduceMean chain -> LayerNorm).
        # Running the generic quant_pre_process first rewrites the graph so the
        # Gelu pattern no longer matches. On whisper-tiny that left 6 bare Erf
        # nodes the HTP compiler rejected, each one splitting the NPU graph and
        # forcing a CPU round trip.
        try:
            changed = qnn_preprocess_model(
                str(input_path),
                str(preprocessed_path),
                fuse_layernorm=True,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
            )
            if changed:
                logger.info("  QNN preprocessing applied graph changes (Gelu/LayerNorm fusion)")
            else:
                preprocessed_path = input_path
        except Exception as e:
            logger.warning(f"  QNN preprocessing failed ({e}); using original model")
            preprocessed_path = input_path
    else:
        try:
            quant_pre_process(str(input_path), str(preprocessed_path), auto_merge=True)
        except Exception as e:
            logger.warning(
                f"  Pre-processing failed ({e}), using original model. "
                "Quantization may still work but could be less optimal."
            )
            preprocessed_path = input_path

    # Step 2: Quantize
    logger.info("  Step 2/3: Quantizing...")

    if calibrate:
        # Static quantization with calibration data — more accurate
        logger.info(
            f"  Using static quantization with {num_calibration_samples} calibration samples"
        )
        calibration_reader = WhisperCalibrationDataReader(
            str(preprocessed_path),
            num_samples=num_calibration_samples,
            model_id=model_id,
            audio_dir=audio_dir,
        )

        if HAS_QNN_HELPERS:
            # Let ORT choose the encodings QNN actually wants. This is the
            # difference between a model that runs on the NPU and one that
            # loads fine and runs entirely on CPU.
            qnn_config = get_qnn_qdq_config(
                str(preprocessed_path),
                calibration_reader,
                calibrate_method=CalibrationMethod.MinMax,
                activation_type=atype,
                weight_type=wtype,
                per_channel=per_channel,
            )
            # NOTE: config-based quantization goes through quantize(), not
            # quantize_static() — the latter's third positional arg is the
            # calibration reader, so passing a config there fails with
            # "'StaticQuantConfig' object has no attribute 'get_next'".
            quantize(
                str(preprocessed_path),
                str(output_path),
                qnn_config,
            )
        else:
            # Manual fallback. Note activation_type=QUInt8 (asymmetric) — NOT
            # QInt8 symmetric, which is the common wrong answer for QNN.
            quantize_static(
                model_input=str(preprocessed_path),
                model_output=str(output_path),
                calibration_data_reader=calibration_reader,
                quant_format=QuantFormat.QDQ,  # never QOperator for QNN
                weight_type=wtype,
                activation_type=atype,
                per_channel=per_channel,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={
                    "WeightSymmetric": wtype == QuantType.QInt8,  # symmetric only if signed
                    "ActivationSymmetric": False,  # uint8 activations: asymmetric
                    "EnableSubgraph": False,       # QNN does not consume subgraphs
                    "ForceQuantizeNoInputCheck": False,
                    "MatMulConstBOnly": False,
                },
            )
    else:
        # Dynamic quantization produces DynamicQuantizeLinear / MatMulInteger /
        # ConvInteger. QNN EP has no builders for any of them, so the resulting
        # model runs 100% on CPU — while loading cleanly and returning correct
        # outputs. This path exists only for CPU-baseline comparison.
        logger.warning(
            "=" * 70 + "\n"
            "  DYNAMIC QUANTIZATION SELECTED — OUTPUT WILL NOT RUN ON THE NPU\n"
            "  Emits DynamicQuantizeLinear / MatMulInteger / ConvInteger, which\n"
            "  QNN EP cannot consume. The model will load, return correct results,\n"
            "  and execute entirely on CPU with no error.\n"
            "  Use this only for a CPU baseline. For NPU deployment, drop\n"
            "  --no-calibrate and let static QDQ quantization run.\n"
            + "=" * 70
        )
        quantize_dynamic(
            model_input=str(preprocessed_path),
            model_output=str(output_path),
            weight_type=wtype,
            per_channel=per_channel,
        )

    elapsed = time.time() - start_time

    # Step 3: Verify output
    logger.info("  Step 3/3: Verifying output...")

    if not output_path.exists():
        raise RuntimeError(f"Quantization produced no output at {output_path}")

    output_size_mb = _model_size_mb(output_path)
    compression_ratio = input_size_mb / output_size_mb if output_size_mb > 0 else 0

    # Count QDQ nodes in the quantized model
    import onnx
    quantized_model = onnx.load(str(output_path))
    qdq_count = sum(
        1 for node in quantized_model.graph.node
        if node.op_type in ("QuantizeLinear", "DequantizeLinear")
    )
    total_nodes = len(quantized_model.graph.node)

    # Verify the output is in a format QNN EP can actually consume. Catching this
    # here — rather than on a Snapdragon device weeks later — is the whole point.
    try:
        from scanner.op_registry import detect_quantization_format_error
        op_types = {node.op_type for node in quantized_model.graph.node}
        format_error = detect_quantization_format_error(op_types)
    except ImportError:
        format_error = None

    if format_error:
        logger.error(
            "\n" + "=" * 70 + "\n"
            f"  OUTPUT IS NOT QNN-COMPATIBLE: {format_error['error']}\n"
            f"  Offending ops: {', '.join(format_error['offending_ops'])}\n"
            f"  {format_error['verdict']}\n"
            + "=" * 70
        )
    elif qdq_count == 0:
        logger.warning(
            "  No QuantizeLinear/DequantizeLinear nodes in the output. QNN EP "
            "builds from QDQ node units — a graph with none of them gives it "
            "nothing to claim. Verify with: python -m scanner --input <dir>"
        )
    else:
        logger.info(f"  Format OK: {qdq_count} QDQ nodes — QNN EP can build from this")

    # Clean up preprocessed file
    shutil.rmtree(work_dir, ignore_errors=True)

    result = {
        "input_model": str(input_path.resolve()),
        "output_model": str(output_path.resolve()),
        "quant_format": quant_format,
        "weight_type": weight_type,
        "activation_type": activation_type.upper(),
        "calibrated": calibrate,
        "per_channel": per_channel,
        "input_size_mb": round(input_size_mb, 2),
        "output_size_mb": round(output_size_mb, 2),
        "compression_ratio": round(compression_ratio, 2),
        "total_nodes": total_nodes,
        "qdq_nodes": qdq_count,
        "qnn_compatible": format_error is None and qdq_count > 0,
        "format_error": format_error,
        "quantization_time_seconds": round(elapsed, 2),
    }

    logger.info(f"\nQuantization complete in {elapsed:.1f}s:")
    logger.info(f"  Input:       {input_size_mb:.1f} MB")
    logger.info(f"  Output:      {output_size_mb:.1f} MB ({compression_ratio:.1f}x compression)")
    logger.info(f"  QDQ nodes:   {qdq_count} / {total_nodes} total nodes")

    return result


def quantize_whisper_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    weight_type: str = "UINT8",
    calibrate: bool = True,
    num_calibration_samples: int = 50,
    model_id: str | None = None,
    audio_dir: str | Path | None = None,
    activation_type: str = "UINT16",
) -> dict:
    """
    Quantize all ONNX files in a Whisper export directory.

    Whisper exports as multiple files — we quantize each separately:
      - encoder_model.onnx       → encoder_model.onnx (quantized)
      - decoder_model.onnx       → decoder_model.onnx (quantized)
      - decoder_with_past_model.onnx → decoder_with_past_model.onnx (quantized)

    Args:
        input_dir:               Directory containing exported ONNX files
        output_dir:              Directory for quantized output files
        weight_type:             "INT8" or "UINT8"
        calibrate:               Whether to use calibration data
        num_calibration_samples: Number of calibration samples

    Returns:
        dict with results for each quantized model component
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Find all ONNX files to quantize
    onnx_files = list(input_path.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files found in {input_path}")

    # The exporter records which HF checkpoint this came from; the feature
    # extractor must match it or the calibration features are subtly wrong.
    if model_id is None:
        model_id = "openai/whisper-tiny"
        config_file = input_path / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as cf:
                    model_id = json.load(cf).get("model_id", model_id)
            except Exception as e:
                logger.debug(f"Could not read model_id from config.json: {e}")
        logger.info(f"Calibration feature extractor: {model_id}")

    logger.info(f"Found {len(onnx_files)} ONNX files to quantize in {input_path}")

    results = {}
    overall_start = time.time()

    for onnx_file in onnx_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {onnx_file.name}")
        logger.info(f"{'='*60}")

        output_file = output_path / onnx_file.name

        try:
            result = quantize_onnx_model(
                input_model_path=onnx_file,
                output_model_path=output_file,
                quant_format="QDQ",
                weight_type=weight_type,
                calibrate=calibrate,
                num_calibration_samples=num_calibration_samples,
                model_id=model_id,
                audio_dir=audio_dir,
                activation_type=activation_type,
            )
            results[onnx_file.name] = result
        except Exception as e:
            logger.error(f"Failed to quantize {onnx_file.name}: {e}")
            results[onnx_file.name] = {"error": str(e)}

    # Copy non-ONNX files (config.json, tokenizer, etc.)
    import shutil
    for f in input_path.iterdir():
        if not f.is_file() or f.suffix == ".onnx":
            continue
        # Never copy external-weight sidecars. They belong to the FP32 input; the
        # quantizer writes its own weights (usually inline). Copying them leaves a
        # stale multi-MB orphan next to the quantized model that nothing reads and
        # that makes the output directory look like the quantization did nothing.
        if f.name.endswith(".onnx.data") or f.suffix == ".data":
            logger.debug(f"  Skipped external-data sidecar: {f.name}")
            continue
        dest = output_path / f.name
        if not dest.exists():
            shutil.copy2(f, dest)
            logger.info(f"  Copied: {f.name}")

    overall_elapsed = time.time() - overall_start

    summary = {
        "input_dir": str(input_path.resolve()),
        "output_dir": str(output_path.resolve()),
        "weight_type": weight_type,
        "activation_type": activation_type.upper(),
        "calibrated": calibrate,
        "total_time_seconds": round(overall_elapsed, 2),
        "models": results,
    }

    # Save summary as JSON for the scanner to consume
    summary_path = output_path / "quantization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Hexagon Bridge — INT8 QDQ Quantization for QNN EP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quantize entire Whisper export (encoder + decoder + decoder_with_past)
  python -m converter.quantize --input ./models/whisper-medium-onnx --output ./models/whisper-medium-int8

  # Quantize a single ONNX file
  python -m converter.quantize --input ./models/whisper-medium-onnx/encoder_model.onnx --output ./models/encoder_int8.onnx

  # Dynamic quantization (faster, no calibration needed)
  python -m converter.quantize --input ./models/whisper-medium-onnx --output ./models/whisper-medium-int8 --no-calibrate

  # INT4 quantization (more aggressive compression)
  python -m converter.quantize --input ./models/whisper-medium-onnx --output ./models/whisper-medium-int4 --weight-type UINT8
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input ONNX file or directory of ONNX files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for quantized model(s)",
    )
    parser.add_argument(
        "--weight-type",
        type=str,
        choices=["UINT8", "INT8"],
        default="UINT8",
        help="Weight quantization type (default: UINT8 — HTP LayerNorm rejects signed int8 gamma under a16)",
    )
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip calibration (use dynamic quantization instead)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=50,
        help="Number of calibration samples (default: 50)",
    )
    parser.add_argument(
        "--activation-type",
        type=str,
        choices=["UINT16", "UINT8"],
        default="UINT16",
        help="Activation precision. UINT16 (default) is required for transformer "
             "accuracy on HTP; UINT8 measured cosine 0.55 vs FP32 on whisper-tiny.",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory of real audio files (.wav/.flac/.mp3/.ogg) for calibration. "
             "Strongly recommended — synthetic audio is a fallback, not a substitute.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HF model id for the feature extractor (default: read from config.json)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)

    if input_path.is_dir():
        # Quantize all ONNX files in directory
        result = quantize_whisper_pipeline(
            input_dir=input_path,
            output_dir=args.output,
            weight_type=args.weight_type,
            calibrate=not args.no_calibrate,
            num_calibration_samples=args.calibration_samples,
            model_id=args.model_id,
            audio_dir=args.audio_dir,
            activation_type=args.activation_type,
        )
    elif input_path.is_file() and input_path.suffix == ".onnx":
        # Quantize single file
        result = quantize_onnx_model(
            input_model_path=input_path,
            output_model_path=args.output,
            quant_format="QDQ",
            weight_type=args.weight_type,
            calibrate=not args.no_calibrate,
            num_calibration_samples=args.calibration_samples,
            model_id=args.model_id or "openai/whisper-tiny",
            audio_dir=args.audio_dir,
            activation_type=args.activation_type,
        )
    else:
        logger.error(f"Input must be a .onnx file or directory: {input_path}")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 60)
    print("QUANTIZATION SUMMARY")
    print("=" * 60)
    if "models" in result:
        for name, info in result["models"].items():
            if "error" in info:
                print(f"  ✗ {name}: FAILED — {info['error']}")
            else:
                print(
                    f"  ✓ {name}: {info['input_size_mb']} MB → {info['output_size_mb']} MB "
                    f"({info['compression_ratio']}x), {info['qdq_nodes']} QDQ nodes"
                )
    else:
        print(
            f"  ✓ {input_path.name}: {result['input_size_mb']} MB → {result['output_size_mb']} MB "
            f"({result['compression_ratio']}x), {result['qdq_nodes']} QDQ nodes"
        )
    print(f"  Total time: {result.get('total_time_seconds', result.get('quantization_time_seconds', 'N/A'))}s")
    print("=" * 60)
    print("\nNext step: Scan QNN operator coverage:")
    print(
        f"  python -m scanner --input {args.output}"
    )


if __name__ == "__main__":
    main()
