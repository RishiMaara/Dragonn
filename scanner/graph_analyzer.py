"""
Hexagon Bridge — ONNX Graph Analyzer
=====================================
Walks the ONNX computational graph node-by-node and checks each operator
against the QNN EP supported operator registry.

This is the core deliverable — the tool that answers:
  "What percentage of this model can actually run on the NPU?"

Usage:
    python -m scanner --input ./models/whisper-medium-int8/encoder_model.onnx
    python -m scanner --input ./models/whisper-medium-int8/ --all
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hexagon-bridge.scanner")


@dataclass
class OpAnalysis:
    """Analysis result for a single operator instance in the graph."""
    op_type: str
    node_name: str
    supported: bool
    support_level: str      # "full", "partial", "quantized", "unsupported", "unknown"
    category: str
    notes: str
    input_shapes: list[str] = field(default_factory=list)
    output_shapes: list[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    """Complete coverage analysis for an ONNX model."""
    model_path: str
    model_name: str
    total_nodes: int
    supported_nodes: int
    unsupported_nodes: int
    coverage_percent: float
    
    # Breakdown by support level
    full_support_count: int = 0
    partial_support_count: int = 0
    unsupported_count: int = 0
    unknown_count: int = 0
    
    # Breakdown by category
    category_coverage: dict = field(default_factory=dict)
    
    # The critical info: exactly which ops fall back to CPU
    fallback_ops: list[dict] = field(default_factory=list)
    
    # All unique op types found
    op_type_counts: dict = field(default_factory=dict)

    # Whole-model verdict: model quantized in a format QNN EP cannot consume.
    # When set, coverage_percent is meaningless — nothing runs on the NPU.
    format_error: Optional[dict] = None

    # Ground truth from the real HTP compiler (scanner/htp_compile.py), when run.
    htp_compile: Optional[dict] = None

    # Per-node analysis (for detailed reports)
    node_analyses: list[OpAnalysis] = field(default_factory=list)

    @property
    def effective_coverage_percent(self) -> float:
        """
        Coverage after accounting for whole-model format errors.

        A graph can look 85% NPU-eligible node-by-node and still run entirely on
        CPU because its quantization format is one QNN EP cannot build from.
        This is the number to report.
        """
        return 0.0 if self.format_error else self.coverage_percent

    def to_dict(self) -> dict:
        """Convert to serializable dict (excluding detailed node analyses)."""
        d = asdict(self)
        # Remove the detailed per-node data from the summary
        d.pop("node_analyses", None)
        return d


_ONNX_DTYPES = {2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32"}


def _dequantized_dtype(tensor: str, producer: dict, initializers: dict) -> str | None:
    """
    The quantized dtype behind a float tensor, or None if it isn't dequantized.

    Weights: DequantizeLinear(initializer). Activations: DequantizeLinear fed by
    QuantizeLinear. Either way the zero point, when present, carries the dtype.
    """
    dq = producer.get(tensor)
    if dq is None or dq.op_type != "DequantizeLinear":
        return None

    candidates = [dq.input[2]] if len(dq.input) > 2 and dq.input[2] else []
    candidates.append(dq.input[0])
    for name in candidates:
        if name in initializers:
            return _ONNX_DTYPES.get(initializers[name].data_type)

    q = producer.get(dq.input[0])
    if (q is not None and q.op_type == "QuantizeLinear"
            and len(q.input) > 2 and q.input[2] in initializers):
        return _ONNX_DTYPES.get(initializers[q.input[2]].data_type)
    return None


def analyze_onnx_model(
    model_path: str | Path,
    include_node_details: bool = False,
) -> CoverageReport:
    """
    Analyze an ONNX model's computational graph for QNN EP compatibility.

    Walks every node in the graph, checks it against the QNN supported
    operator registry, and produces a detailed coverage report.

    Args:
        model_path:            Path to the ONNX model file
        include_node_details:  If True, include per-node analysis in the report

    Returns:
        CoverageReport with complete analysis
    """
    from scanner.op_registry import (
        is_supported,
        get_category,
        detect_quantization_format_error,
        check_quant_constraints,
    )

    try:
        import onnx
        from onnx import shape_inference
    except ImportError:
        logger.error("onnx package not installed. Run: pip install onnx")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Analyzing: {model_path.name}")

    # Load and optionally run shape inference
    model = onnx.load(str(model_path))
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning(f"Shape inference failed ({e}), proceeding without it")

    # Build a map of tensor names to shapes for reporting
    tensor_shapes = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        shape = []
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.HasField("shape"):
            for dim in vi.type.tensor_type.shape.dim:
                if dim.dim_value > 0:
                    shape.append(str(dim.dim_value))
                elif dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append("?")
        tensor_shapes[vi.name] = shape

    # The HTP backend validates each node's quantization encodings, not just its
    # op type — so resolve every tensor's quantized dtype through its
    # DequantizeLinear. This is what catches a signed-int8 LayerNorm gamma.
    producer = {out: n for n in model.graph.node for out in n.output}
    initializers = {i.name: i for i in model.graph.initializer}

    # Analyze each node
    node_analyses: list[OpAnalysis] = []
    op_type_counter = Counter()
    category_stats = defaultdict(lambda: {"total": 0, "supported": 0})
    fallback_ops = []

    for node in model.graph.node:
        op_type = node.op_type
        node_name = node.name or f"unnamed_{op_type}_{len(node_analyses)}"

        # Skip QDQ nodes in the coverage count — they're scaffolding, not compute
        # But still count them for completeness
        is_qdq = op_type in ("QuantizeLinear", "DequantizeLinear")

        supported, support_level, notes = is_supported(op_type)
        category = get_category(op_type)

        if supported and not is_qdq:
            input_qtypes = [_dequantized_dtype(t, producer, initializers) for t in node.input]
            rejection = check_quant_constraints(op_type, input_qtypes)
            if rejection:
                supported, support_level, notes = False, "rejected", rejection

        # Get shapes for context
        input_shapes = [
            f"{name}:[{','.join(tensor_shapes.get(name, ['?']))}]"
            for name in node.input
            if name  # skip empty optional inputs
        ]
        output_shapes = [
            f"{name}:[{','.join(tensor_shapes.get(name, ['?']))}]"
            for name in node.output
            if name
        ]

        analysis = OpAnalysis(
            op_type=op_type,
            node_name=node_name,
            supported=supported,
            support_level=support_level,
            category=category,
            notes=notes,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
        )

        node_analyses.append(analysis)
        op_type_counter[op_type] += 1

        # Track category stats (exclude QDQ scaffolding)
        if not is_qdq:
            category_stats[category]["total"] += 1
            if supported:
                category_stats[category]["supported"] += 1

        # Track fallback ops
        if not supported and not is_qdq:
            # Group by op_type — we don't need 50 entries for the same op
            existing = next(
                (f for f in fallback_ops if f["op_type"] == op_type), None
            )
            if existing:
                existing["count"] += 1
            else:
                fallback_ops.append({
                    "op_type": op_type,
                    "support_level": support_level,
                    "reason": notes,
                    "count": 1,
                    "category": category,
                    "example_node": node_name,
                })

    # Calculate overall coverage (excluding QDQ nodes)
    compute_nodes = [
        a for a in node_analyses
        if a.op_type not in ("QuantizeLinear", "DequantizeLinear")
    ]
    total_compute = len(compute_nodes)
    supported_compute = sum(1 for a in compute_nodes if a.supported)
    unsupported_compute = total_compute - supported_compute

    coverage_pct = (
        (supported_compute / total_compute * 100) if total_compute > 0 else 0.0
    )

    # Category coverage breakdown
    category_coverage = {}
    for cat, stats in sorted(category_stats.items()):
        pct = (
            (stats["supported"] / stats["total"] * 100)
            if stats["total"] > 0
            else 0.0
        )
        category_coverage[cat] = {
            "total": stats["total"],
            "supported": stats["supported"],
            "unsupported": stats["total"] - stats["supported"],
            "coverage_percent": round(pct, 1),
        }

    # Support level counts
    full_count = sum(1 for a in compute_nodes if a.support_level == "full")
    partial_count = sum(1 for a in compute_nodes if a.support_level == "partial")
    unsupported_count = sum(
        1 for a in compute_nodes
        if a.support_level in ("unsupported", "unknown", "format_error", "rejected")
    )

    # Whole-model check: is this even in a format QNN EP can consume?
    # This overrides the node-level percentage — a graph full of NPU-friendly ops
    # still runs entirely on CPU if it was quantized the wrong way.
    format_error = detect_quantization_format_error(op_type_counter.keys())

    # Sort fallback ops by count (most impactful first)
    fallback_ops.sort(key=lambda f: f["count"], reverse=True)

    report = CoverageReport(
        model_path=str(model_path.resolve()),
        model_name=model_path.stem,
        total_nodes=total_compute,
        supported_nodes=supported_compute,
        unsupported_nodes=unsupported_compute,
        coverage_percent=round(coverage_pct, 1),
        full_support_count=full_count,
        partial_support_count=partial_count,
        unsupported_count=unsupported_count,
        unknown_count=sum(
            1 for a in compute_nodes if a.support_level == "unknown"
        ),
        category_coverage=category_coverage,
        fallback_ops=fallback_ops,
        op_type_counts=dict(op_type_counter.most_common()),
        format_error=format_error,
        node_analyses=node_analyses if include_node_details else [],
    )

    if format_error:
        logger.error(
            f"  QUANTIZATION FORMAT ERROR: {format_error['error']}\n"
            f"    Offending ops: {', '.join(format_error['offending_ops'])}\n"
            f"    {format_error['verdict']}\n"
            f"    Node-level coverage reads {coverage_pct:.1f}%, but effective "
            f"NPU coverage is 0% — QNN EP will claim none of this graph."
        )
    else:
        logger.info(
            f"  Coverage: {coverage_pct:.1f}% NPU-eligible "
            f"({supported_compute}/{total_compute} compute nodes)"
        )
    if fallback_ops:
        logger.info(
            f"  Fallback ops: {', '.join(f['op_type'] for f in fallback_ops[:5])}"
        )

    return report


def analyze_directory(
    model_dir: str | Path,
    include_node_details: bool = False,
) -> dict[str, CoverageReport]:
    """
    Analyze all ONNX models in a directory.

    Returns:
        dict mapping filename to CoverageReport
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {model_dir}")

    onnx_files = sorted(model_dir.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No ONNX files found in {model_dir}")

    reports = {}
    for onnx_file in onnx_files:
        try:
            report = analyze_onnx_model(onnx_file, include_node_details)
            reports[onnx_file.name] = report
        except Exception as e:
            logger.error(f"Failed to analyze {onnx_file.name}: {e}")

    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Hexagon Bridge — QNN Operator Coverage Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single ONNX model
  python -m scanner --input ./models/whisper-medium-int8/encoder_model.onnx

  # Analyze all ONNX files in a directory
  python -m scanner --input ./models/whisper-medium-int8/

  # Save detailed JSON report
  python -m scanner --input ./models/whisper-medium-int8/ --json ./reports/coverage.json

  # Include per-node details
  python -m scanner --input ./models/whisper-medium-int8/ --detailed
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="ONNX model file or directory of ONNX files",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Save detailed JSON report to this path",
    )
    parser.add_argument(
        "--svg",
        type=str,
        default=None,
        help="Also save the colored report as an SVG image (e.g. for a README)",
    )
    parser.add_argument(
        "--compile-check",
        action="store_true",
        help="Also compile with the real HTP compiler (needs onnxruntime-qnn; works on x64)",
    )
    parser.add_argument(
        "--htp-arch",
        default="73",
        help="Hexagon arch for --compile-check (73 = Snapdragon X Elite / X Plus)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include per-node analysis in the report",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # The report uses box-drawing characters, em-dashes and status emoji. A
    # default Windows console is cp1252 and raises UnicodeEncodeError on all of
    # them — which would crash the scanner precisely when it has bad news to
    # deliver. Force UTF-8 before anything prints.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)

    if input_path.is_file() and input_path.suffix == ".onnx":
        report = analyze_onnx_model(input_path, args.detailed)
        reports = {input_path.name: report}
    elif input_path.is_dir():
        reports = analyze_directory(input_path, args.detailed)
    else:
        logger.error(f"Input must be an .onnx file or directory: {input_path}")
        sys.exit(1)

    if args.compile_check:
        from scanner.htp_compile import compile_check
        for filename, report in reports.items():
            logger.info(f"Compiling {filename} with the local HTP compiler...")
            report.htp_compile = compile_check(report.model_path, htp_arch=args.htp_arch)

    # Import and use the report formatter
    from scanner.report import print_coverage_report, save_json_report

    console = None
    if args.svg:
        from rich.console import Console
        console = Console(record=True, width=100)

    for filename, report in reports.items():
        print_coverage_report(report, console=console)

    if args.svg:
        console.save_svg(args.svg, title=f"python -m scanner --input {input_path.name}")
        print(f"\nSVG report saved to: {args.svg}")

    if args.json:
        save_json_report(reports, args.json)
        print(f"\nJSON report saved to: {args.json}")


if __name__ == "__main__":
    main()
