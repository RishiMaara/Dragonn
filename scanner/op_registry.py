"""
Hexagon Bridge — QNN Supported Operator Registry
=================================================
Maintains the list of ONNX operators supported by the QNN Execution Provider
on Snapdragon X (Hexagon NPU).

This is the source of truth for our coverage scanner. Operators not in this
list will fall back to CPU EP at runtime — which is exactly what we want to
detect and report BEFORE deployment.

Sources:
  - QNN SDK documentation (Qualcomm)
  - ONNX Runtime QNN EP supported operators list
  - https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
  - Empirical testing on Snapdragon X Elite devices

Last updated: 2026-09 (QNN SDK 2.28+, ONNX Runtime 1.18+)
"""

# ============================================================================
# QNN EP Supported ONNX Operators
# ============================================================================
# Each entry maps an ONNX op_type to its support status and any constraints.
#
# Support levels:
#   "full"      — Fully supported on Hexagon NPU via QNN
#   "partial"   — Supported with constraints (see 'constraints' field)
#   "quantized" — Only supported when inputs are quantized (INT8/UINT8)
#   "fp16"      — Only supported in FP16 mode
# ============================================================================

QNN_SUPPORTED_OPS: dict[str, dict] = {
    # ── Activation Functions ──
    "Relu": {
        "support": "full",
        "category": "activation",
        "notes": "Natively supported, fused with Conv/MatMul when possible",
    },
    "Sigmoid": {
        "support": "full",
        "category": "activation",
    },
    "Tanh": {
        "support": "full",
        "category": "activation",
    },
    "LeakyRelu": {
        "support": "full",
        "category": "activation",
    },
    "Elu": {
        "support": "full",
        "category": "activation",
    },
    "HardSigmoid": {
        "support": "full",
        "category": "activation",
    },
    "HardSwish": {
        "support": "full",
        "category": "activation",
    },
    "Gelu": {
        "support": "partial",
        "category": "activation",
        "constraints": "Approximate GELU only; exact GELU may decompose",
        "min_opset": 20,
    },
    "Selu": {
        "support": "full",
        "category": "activation",
    },
    "Softplus": {
        "support": "full",
        "category": "activation",
    },
    "Clip": {
        "support": "full",
        "category": "activation",
        "notes": "Used for ReLU6 patterns",
    },

    # ── Convolution & Pooling ──
    "Conv": {
        "support": "full",
        "category": "convolution",
        "notes": "1D, 2D, 3D. Core NPU operation with highest efficiency",
    },
    "ConvTranspose": {
        "support": "full",
        "category": "convolution",
    },
    "AveragePool": {
        "support": "full",
        "category": "pooling",
    },
    "GlobalAveragePool": {
        "support": "full",
        "category": "pooling",
    },
    "MaxPool": {
        "support": "full",
        "category": "pooling",
    },
    "GlobalMaxPool": {
        "support": "full",
        "category": "pooling",
    },

    # ── Linear Algebra ──
    "MatMul": {
        "support": "full",
        "category": "linear",
        "notes": "Core NPU operation. Fused with QDQ for INT8 acceleration",
    },
    "Gemm": {
        "support": "full",
        "category": "linear",
    },

    # ── Element-wise Operations ──
    "Add": {
        "support": "full",
        "category": "elementwise",
    },
    "Sub": {
        "support": "full",
        "category": "elementwise",
    },
    "Mul": {
        "support": "full",
        "category": "elementwise",
    },
    "Div": {
        "support": "full",
        "category": "elementwise",
    },
    "Pow": {
        "support": "partial",
        "category": "elementwise",
        "constraints": "Only integer exponents in some cases",
    },
    "Sqrt": {
        "support": "full",
        "category": "elementwise",
    },
    "Reciprocal": {
        "support": "full",
        "category": "elementwise",
    },
    "Neg": {
        "support": "full",
        "category": "elementwise",
    },
    "Abs": {
        "support": "full",
        "category": "elementwise",
    },
    "Exp": {
        "support": "full",
        "category": "elementwise",
    },
    "Log": {
        "support": "full",
        "category": "elementwise",
    },
    "Floor": {
        "support": "full",
        "category": "elementwise",
    },
    "Ceil": {
        "support": "full",
        "category": "elementwise",
    },
    "Round": {
        "support": "full",
        "category": "elementwise",
    },
    "Min": {
        "support": "full",
        "category": "elementwise",
    },
    "Max": {
        "support": "full",
        "category": "elementwise",
    },
    "Equal": {
        "support": "full",
        "category": "elementwise",
    },
    "Greater": {
        "support": "full",
        "category": "elementwise",
    },
    "Less": {
        "support": "full",
        "category": "elementwise",
    },
    "And": {
        "support": "full",
        "category": "elementwise",
    },
    "Or": {
        "support": "full",
        "category": "elementwise",
    },
    "Not": {
        "support": "full",
        "category": "elementwise",
    },
    "Where": {
        "support": "full",
        "category": "elementwise",
    },
    "Erf": {
        "support": "partial",
        "category": "elementwise",
        "constraints": (
            "Rejected by the HTP compiler under 16-bit activations (verified: QNN "
            "2.50, a16w8 whisper-tiny — 6 Erf nodes left on CPU, each splitting the "
            "NPU graph). Appears when GELU is exported decomposed. Fix: run "
            "qnn_preprocess_model on the raw export, which fuses the Erf chain into "
            "a native Gelu — and do not run quant_pre_process before it, which "
            "rewrites the pattern so the fusion no longer matches."
        ),
    },

    # ── Normalization ──
    "BatchNormalization": {
        "support": "full",
        "category": "normalization",
        "notes": "Fused with Conv when possible",
    },
    "InstanceNormalization": {
        "support": "full",
        "category": "normalization",
    },
    "LayerNormalization": {
        "support": "partial",
        "category": "normalization",
        "constraints": (
            "Supported in QNN SDK 2.20+, but the HTP backend rejects a SIGNED int8 "
            "gamma under 16-bit activations (backendValidateOpConfig error 3110). "
            "Verified on a real Snapdragon X Elite (QNN 2.45) and locally (QNN 2.50): "
            "a16 + int8 weights left all 9 whisper-tiny LayerNorms off-NPU; a16 + "
            "uint8 weights compiled the whole encoder as one NPU graph."
        ),
        "min_qnn_sdk": "2.20",
    },
    "GroupNormalization": {
        "support": "partial",
        "category": "normalization",
        "constraints": "Supported in newer QNN SDK versions only",
        "min_qnn_sdk": "2.24",
    },

    # ── Reduction ──
    "ReduceMean": {
        "support": "full",
        "category": "reduction",
    },
    "ReduceSum": {
        "support": "full",
        "category": "reduction",
    },
    "ReduceMax": {
        "support": "full",
        "category": "reduction",
    },
    "ReduceMin": {
        "support": "full",
        "category": "reduction",
    },
    "ReduceProd": {
        "support": "full",
        "category": "reduction",
    },
    "ReduceL2": {
        "support": "full",
        "category": "reduction",
    },
    "ArgMax": {
        "support": "full",
        "category": "reduction",
    },
    "ArgMin": {
        "support": "full",
        "category": "reduction",
    },

    # ── Tensor Manipulation ──
    "Reshape": {
        "support": "full",
        "category": "tensor",
        "notes": "Zero-cost on NPU (metadata only, no data movement)",
    },
    "Transpose": {
        "support": "full",
        "category": "tensor",
    },
    "Squeeze": {
        "support": "full",
        "category": "tensor",
    },
    "Unsqueeze": {
        "support": "full",
        "category": "tensor",
    },
    "Flatten": {
        "support": "full",
        "category": "tensor",
    },
    "Concat": {
        "support": "full",
        "category": "tensor",
    },
    "Split": {
        "support": "full",
        "category": "tensor",
    },
    "Slice": {
        "support": "full",
        "category": "tensor",
    },
    "Gather": {
        "support": "full",
        "category": "tensor",
    },
    "GatherElements": {
        "support": "partial",
        "category": "tensor",
        "constraints": "Limited axis support",
    },
    "GatherND": {
        "support": "partial",
        "category": "tensor",
        "constraints": "Limited batch dimensions",
    },
    "Scatter": {
        "support": "partial",
        "category": "tensor",
        "constraints": "ScatterND has limited support",
    },
    "ScatterElements": {
        "support": "partial",
        "category": "tensor",
    },
    "ScatterND": {
        "support": "partial",
        "category": "tensor",
    },
    "Pad": {
        "support": "full",
        "category": "tensor",
    },
    "Tile": {
        "support": "full",
        "category": "tensor",
    },
    "Expand": {
        "support": "full",
        "category": "tensor",
    },
    "Shape": {
        "support": "full",
        "category": "tensor",
        "notes": "Compile-time constant — no runtime cost",
    },
    "ConstantOfShape": {
        "support": "full",
        "category": "tensor",
    },
    "Identity": {
        "support": "full",
        "category": "tensor",
    },
    "Cast": {
        "support": "partial",
        "category": "tensor",
        "constraints": "Limited to supported dtype pairs",
    },
    "Resize": {
        "support": "partial",
        "category": "tensor",
        "constraints": "Nearest and bilinear modes supported; bicubic may fall back",
    },

    # ── Quantization ──
    "QuantizeLinear": {
        "support": "full",
        "category": "quantization",
        "notes": "Core QDQ node — always handled by QNN EP",
    },
    "DequantizeLinear": {
        "support": "full",
        "category": "quantization",
        "notes": "Core QDQ node — always handled by QNN EP",
    },
    # NOTE: QLinearConv / QLinearMatMul are deliberately NOT listed here.
    # Those are QOperator-format ops. QNN EP builds its graph from QDQ node
    # units (DequantizeLinear -> Op -> QuantizeLinear); it has no builders for
    # the QOperator family. See QOPERATOR_OPS below.

    # ── Attention / Transformer Components ──
    "Softmax": {
        "support": "partial",
        "category": "attention",
        "constraints": (
            "Supported on axis=-1 for most shapes. Large attention matrices "
            "may exceed NPU memory and fall back to CPU. This is the most "
            "common fallback op in transformer models."
        ),
    },
    "Attention": {
        "support": "partial",
        "category": "attention",
        "constraints": (
            "Fused attention (com.microsoft domain) has limited support. "
            "Decomposed attention (separate Q/K/V MatMul + Softmax) preferred."
        ),
    },
    "MultiHeadAttention": {
        "support": "partial",
        "category": "attention",
        "constraints": "Limited support; decomposed form preferred for QNN",
    },

    # ── Recurrent ──
    "LSTM": {
        "support": "partial",
        "category": "recurrent",
        "constraints": "Unidirectional only; limited hidden sizes",
    },
    "GRU": {
        "support": "partial",
        "category": "recurrent",
        "constraints": "Limited support",
    },
}

# ============================================================================
# QUANTIZATION FORMAT ERRORS
# ============================================================================
# These two sets are special. Their presence in a graph is not "one op fell
# back to CPU" — it means the model was quantized in a format QNN EP cannot
# consume at all, and the ENTIRE graph will run on CPU.
#
# QNN EP builds its graph from QDQ node units:
#     DequantizeLinear -> Op -> QuantizeLinear
#
# It has no builders for either alternative encoding below. A model containing
# these ops loads fine, produces correct outputs, and never touches the NPU —
# which is exactly the silent failure this scanner exists to catch.
# ============================================================================

# Emitted by onnxruntime.quantization.quantize_dynamic().
# Activation ranges are computed at runtime, which a fixed-point NPU cannot do.
DYNAMIC_QUANT_OPS: set[str] = {
    "DynamicQuantizeLinear",
    "DynamicQuantizeMatMul",
    "DynamicQuantizeLSTM",
    "MatMulInteger",
    "MatMulIntegerToFloat",
    "ConvInteger",
}

# Emitted by quantize_static(quant_format=QuantFormat.QOperator).
# Fused integer ops rather than QDQ node units.
QOPERATOR_OPS: set[str] = {
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "QLinearMul",
    "QLinearAveragePool",
    "QLinearGlobalAveragePool",
    "QLinearSigmoid",
    "QLinearSoftmax",
    "QLinearLeakyRelu",
    "QLinearConcat",
    "QGemm",
}

# ============================================================================
# Operators known to ALWAYS fall back to CPU on QNN EP
# These are the ops that will show up in our coverage report as "CPU-only"
# ============================================================================

QNN_UNSUPPORTED_OPS: set[str] = {
    # Custom / Microsoft domain ops
    "EmbedLayerNormalization",
    "SkipLayerNormalization",
    "BiasGelu",
    "FastGelu",
    "FusedMatMul",
    "Attention",                    # com.microsoft.Attention specifically
    "BeamSearch",
    "GreedySearch",
    "Sampling",

    # Complex ops that decompose
    "Einsum",
    "CumSum",
    "Trilu",
    "BitShift",
    "StringNormalizer",
    "TfIdfVectorizer",
    "SequenceConstruct",
    "SequenceAt",
    "SequenceLength",
    "ConcatFromSequence",
    "SplitToSequence",

    # Control flow — always CPU
    "Loop",
    "If",
    "Scan",

    # Optional / uncommon
    "NonZero",
    "NonMaxSuppression",
    "RoiAlign",
    "TopK",
    "Unique",
    "Compress",
    "OneHot",
    "Det",
    "Inverse",
    "IsInf",
    "IsNaN",
    "Mod",
    "ReverseSequence",
    "Range",
}

# Format-error ops are unsupported too — folded in so every lookup path sees them.
QNN_UNSUPPORTED_OPS |= DYNAMIC_QUANT_OPS | QOPERATOR_OPS

# ============================================================================
# Category descriptions for human-readable reports
# ============================================================================

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "activation": "Activation Functions (ReLU, GELU, etc.)",
    "convolution": "Convolution Operations",
    "pooling": "Pooling Operations",
    "linear": "Linear Algebra (MatMul, Gemm)",
    "elementwise": "Element-wise Operations (Add, Mul, etc.)",
    "normalization": "Normalization (LayerNorm, BatchNorm, etc.)",
    "reduction": "Reduction Operations (Sum, Mean, etc.)",
    "tensor": "Tensor Manipulation (Reshape, Transpose, etc.)",
    "quantization": "Quantization Nodes (QDQ pairs)",
    "attention": "Attention / Transformer Components",
    "recurrent": "Recurrent Layers (LSTM, GRU)",
}


def is_supported(op_type: str) -> tuple[bool, str, str]:
    """
    Check if an ONNX operator is supported by QNN EP.

    Returns:
        (is_supported, support_level, notes)
        - is_supported: True if the op can run on NPU (full or partial)
        - support_level: "full", "partial", "quantized", "unsupported"
        - notes: Human-readable explanation of constraints
    """
    if op_type in DYNAMIC_QUANT_OPS:
        return False, "format_error", (
            f"'{op_type}' is a DYNAMIC quantization op. QNN EP cannot consume it — "
            "it needs static QDQ node units. The whole graph will run on CPU. "
            "Re-quantize with quantize_static() using ORT's QNN config helpers."
        )

    if op_type in QOPERATOR_OPS:
        return False, "format_error", (
            f"'{op_type}' is a QOperator-format op. QNN EP builds from QDQ node "
            "units only. The whole graph will run on CPU. Re-quantize with "
            "quant_format=QuantFormat.QDQ."
        )

    if op_type in QNN_UNSUPPORTED_OPS:
        return False, "unsupported", "Always falls back to CPU on QNN EP"

    if op_type in QNN_SUPPORTED_OPS:
        info = QNN_SUPPORTED_OPS[op_type]
        level = info["support"]
        notes = info.get("constraints", info.get("notes", ""))
        return True, level, notes

    # Unknown operator — assume unsupported (conservative)
    return False, "unknown", f"Operator '{op_type}' not in QNN EP registry — likely CPU fallback"


def get_category(op_type: str) -> str:
    """Get the category of an ONNX operator."""
    if op_type in QNN_SUPPORTED_OPS:
        return QNN_SUPPORTED_OPS[op_type].get("category", "other")
    return "other"


def get_all_supported_ops() -> list[str]:
    """Return list of all QNN-supported ONNX op types."""
    return sorted(QNN_SUPPORTED_OPS.keys())


def get_all_unsupported_ops() -> list[str]:
    """Return list of all known-unsupported ONNX op types."""
    return sorted(QNN_UNSUPPORTED_OPS)


def check_quant_constraints(op_type: str, input_qtypes: list) -> str | None:
    """
    Rejections that depend on HOW an op's inputs are quantized, not just on the op.

    The op registry answers "can QNN build this op?" — but the HTP backend also
    validates each node's quantization encodings, and some supported ops are
    rejected under specific ones. Only rules verified against the real HTP
    compiler or a real device belong here.

    Args:
        op_type:      ONNX op type
        input_qtypes: per-input quantized dtype ("uint16", "int8", ...), or None
                      where an input is not quantized / not resolvable

    Returns:
        A rejection reason, or None if no known constraint applies.
    """
    act_16bit = bool(input_qtypes) and input_qtypes[0] in ("uint16", "int16")

    if op_type == "LayerNormalization" and act_16bit and len(input_qtypes) > 1 \
            and input_qtypes[1] == "int8":
        return (
            "HTP rejects LayerNorm with a SIGNED int8 gamma under 16-bit activations "
            "(backendValidateOpConfig error 3110). Verified on a real Snapdragon X "
            "Elite (QNN 2.45) and the local HTP compiler (QNN 2.50). Quantize weights "
            "as uint8: --weight-type UINT8."
        )

    if op_type == "Erf" and act_16bit:
        return (
            "HTP rejects Erf under 16-bit activations (verified, QNN 2.50). It appears "
            "when GELU is exported decomposed: run qnn_preprocess_model on the raw "
            "export to fuse the chain into a native Gelu."
        )

    return None


def detect_quantization_format_error(op_types) -> dict | None:
    """
    Check whether a model was quantized in a format QNN EP cannot consume.

    This is a whole-model verdict, not a per-node one. If it fires, the node-level
    coverage percentage is meaningless — nothing runs on the NPU regardless of how
    NPU-friendly the individual ops look.

    Args:
        op_types: iterable of op_type strings present in the graph

    Returns:
        None if the format is fine, else a dict describing the problem and the fix.
    """
    present = set(op_types)

    found_dynamic = sorted(present & DYNAMIC_QUANT_OPS)
    found_qoperator = sorted(present & QOPERATOR_OPS)

    if not found_dynamic and not found_qoperator:
        return None

    if found_dynamic:
        return {
            "error": "dynamic_quantization",
            "offending_ops": found_dynamic,
            "verdict": "This model will run 100% on CPU. The NPU will not be used.",
            "explanation": (
                "These ops compute quantization ranges at runtime. The Hexagon NPU is "
                "a fixed-point engine — ranges must be baked in ahead of time. QNN EP "
                "has no builders for this op family, so it claims none of the graph."
            ),
            "fix": (
                "Re-quantize with static QDQ:\n"
                "  from onnxruntime.quantization import quantize_static, QuantFormat\n"
                "  from onnxruntime.quantization.execution_providers.qnn import (\n"
                "      get_qnn_qdq_config, qnn_preprocess_model)\n"
                "and supply a real calibration data reader."
            ),
        }

    return {
        "error": "qoperator_format",
        "offending_ops": found_qoperator,
        "verdict": "This model will run 100% on CPU. The NPU will not be used.",
        "explanation": (
            "QOperator format fuses quantization into single integer ops. QNN EP "
            "builds its graph from QDQ node units (DequantizeLinear -> Op -> "
            "QuantizeLinear) and has no builders for the QLinear* family."
        ),
        "fix": "Re-quantize with quant_format=QuantFormat.QDQ (not QuantFormat.QOperator).",
    }
