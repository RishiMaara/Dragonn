"""
Scanner tests. Each one pins a failure this project hit for real:

- dynamic quantization graded "85% NPU-eligible" while running 100% on CPU
- a signed-int8 LayerNorm gamma that passed every static check and failed to
  compile on a real Snapdragon X Elite
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

# Test models carry the IR version the real pipeline's models carry (torch's
# exporter writes 10). Newer onnx releases default to IR 14, which ONNX Runtime
# 1.26 refuses to load — that broke CI when it picked up onnx 1.23.
IR_VERSION = 10

from scanner.graph_analyzer import analyze_onnx_model
from scanner.op_registry import (
    check_quant_constraints,
    detect_quantization_format_error,
    is_supported,
)


# ── Registry ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("op", ["MatMulInteger", "ConvInteger", "DynamicQuantizeLinear"])
def test_dynamic_quant_ops_are_not_npu_eligible(op):
    supported, level, _ = is_supported(op)
    assert not supported and level == "format_error"


@pytest.mark.parametrize("op", ["QLinearConv", "QLinearMatMul"])
def test_qoperator_ops_are_not_npu_eligible(op):
    supported, level, _ = is_supported(op)
    assert not supported and level == "format_error"


def test_format_error_detection():
    assert detect_quantization_format_error(["Conv", "QuantizeLinear"]) is None
    assert detect_quantization_format_error(["MatMulInteger"])["error"] == "dynamic_quantization"
    assert detect_quantization_format_error(["QLinearConv"])["error"] == "qoperator_format"


def test_layernorm_signed_gamma_under_16bit_is_rejected():
    assert check_quant_constraints("LayerNormalization", ["uint16", "int8", "int32"])
    assert check_quant_constraints("LayerNormalization", ["uint16", "uint8", "int32"]) is None
    assert check_quant_constraints("LayerNormalization", ["uint8", "int8", "int32"]) is None


def test_erf_under_16bit_is_rejected():
    assert check_quant_constraints("Erf", ["uint16"])
    assert check_quant_constraints("Erf", ["uint8"]) is None


# ── Analyzer on hand-built graphs ─────────────────────────────────────────────

def _layernorm_model(gamma_dtype: int) -> onnx.ModelProto:
    """uint16-activation QDQ LayerNorm whose gamma is quantized as `gamma_dtype`."""
    np_gamma = {TensorProto.INT8: np.int8, TensorProto.UINT8: np.uint8}[gamma_dtype]
    inits = [
        numpy_helper.from_array(np.array(0.01, np.float32), "x_scale"),
        helper.make_tensor("x_zp", TensorProto.UINT16, [], [32768]),
        numpy_helper.from_array(np.ones(4, np_gamma), "gamma_q"),
        numpy_helper.from_array(np.array(0.02, np.float32), "gamma_scale"),
        helper.make_tensor("gamma_zp", gamma_dtype, [], [0]),
        numpy_helper.from_array(np.zeros(4, np.int32), "beta_q"),
        numpy_helper.from_array(np.array(0.0002, np.float32), "beta_scale"),
        helper.make_tensor("beta_zp", TensorProto.INT32, [], [0]),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zp"], ["x_q"]),
        helper.make_node("DequantizeLinear", ["x_q", "x_scale", "x_zp"], ["x_dq"]),
        helper.make_node("DequantizeLinear", ["gamma_q", "gamma_scale", "gamma_zp"], ["gamma"]),
        helper.make_node("DequantizeLinear", ["beta_q", "beta_scale", "beta_zp"], ["beta"]),
        helper.make_node("LayerNormalization", ["x_dq", "gamma", "beta"], ["y"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes, "ln",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        inits,
    )
    return helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 21)])


def test_analyzer_rejects_signed_gamma_layernorm(tmp_path):
    path = tmp_path / "ln_int8.onnx"
    onnx.save(_layernorm_model(TensorProto.INT8), path)
    report = analyze_onnx_model(path)
    assert [f["op_type"] for f in report.fallback_ops] == ["LayerNormalization"]
    assert report.fallback_ops[0]["support_level"] == "rejected"
    assert report.coverage_percent == 0.0


def test_analyzer_accepts_unsigned_gamma_layernorm(tmp_path):
    path = tmp_path / "ln_uint8.onnx"
    onnx.save(_layernorm_model(TensorProto.UINT8), path)
    report = analyze_onnx_model(path)
    assert report.fallback_ops == []
    assert report.coverage_percent == 100.0


def test_dynamic_quant_graph_reports_zero_effective_coverage(tmp_path):
    """The original bug: this shape of graph was graded 85% NPU-eligible."""
    nodes = [
        helper.make_node("DynamicQuantizeLinear", ["x"], ["xq", "xs", "xz"]),
        helper.make_node("MatMulInteger", ["xq", "w", "xz"], ["acc"]),
        helper.make_node("Cast", ["acc"], ["accf"], to=TensorProto.FLOAT),
        helper.make_node("Mul", ["accf", "xs"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes, "dyn",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(np.ones((4, 4), np.uint8), "w")],
    )
    path = tmp_path / "dyn.onnx"
    onnx.save(helper.make_model(graph, ir_version=IR_VERSION, opset_imports=[helper.make_opsetid("", 17)]), path)

    report = analyze_onnx_model(path)
    assert report.format_error["error"] == "dynamic_quantization"
    assert report.coverage_percent > 0          # node-level still looks partly fine...
    assert report.effective_coverage_percent == 0.0   # ...but nothing runs on the NPU
