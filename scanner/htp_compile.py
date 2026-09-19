"""
Hexagon Bridge — Local HTP Compile Check
========================================
Runs Qualcomm's actual Hexagon (HTP) graph compiler on this machine and reports
how the model really partitions: how many separate NPU graphs it compiles into,
and which ops the compiler refused and left on CPU.

Static analysis answers "is this op supported?". The compiler answers "will
THIS node, with THESE quantization encodings, be accepted?" — which is where the
real failures live. On whisper-tiny the static registry predicted 100% NPU for a
model whose 9 LayerNorms the chip rejected; this check reproduced that rejection
on an x64 laptop in ~10 s.

Works on x64: onnxruntime-qnn ships an x64 HTP backend that compiles context
binaries without executing them. Needs `pip install onnxruntime-qnn`.

Passing locally is necessary, not sufficient: the local QNN SDK version may
differ from a given device's, and graph finalization can still fail on silicon.
Confirm with `python -m scripts.aihub_validate`.
"""

import logging
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger("hexagon-bridge.htp-compile")

# Snapdragon X Elite / X Plus: Hexagon v73, SoC model 60 (as reported by the
# device in AI Hub runtime logs). X2 Elite is a later Hexagon generation.
DEFAULT_HTP_ARCH = "73"
DEFAULT_SOC_MODEL = "60"

_BOUNDARY_OPS = ("QuantizeLinear", "DequantizeLinear")


def local_qnn_version() -> str | None:
    try:
        from onnxruntime_qnn.build_and_package_info import qnn_version
        return qnn_version
    except ImportError:
        return None


def compile_check(
    model_path: str | Path,
    htp_arch: str = DEFAULT_HTP_ARCH,
    soc_model: str = DEFAULT_SOC_MODEL,
) -> dict:
    """
    Compile the model for the HTP and inspect the resulting partitioning.

    Returns a dict with:
      available      False if no QNN EP is installed (nothing else is set)
      ok             True if it compiled into exactly one NPU graph with no
                     compute ops left on CPU
      npu_graphs     number of EPContext partitions (1 is ideal)
      cpu_ops        {op_type: count} of compute ops the compiler left on CPU
      boundary_ops   count of QuantizeLinear/DequantizeLinear outside the NPU
                     graphs (a couple at the model edges is normal; more means
                     data is shuttling between NPU fragments and CPU)
      compile_s      wall-clock compile time
      qnn_version    local QNN SDK version
      error          compiler/session error text, if compilation failed
    """
    from scripts.qnn_ep import compile_only_available, create_session

    result = {"available": compile_only_available(), "qnn_version": local_qnn_version()}
    if not result["available"]:
        result["error"] = "QNN EP not installed — pip install onnxruntime-qnn"
        return result

    import onnx
    import onnxruntime as ort

    work = Path(tempfile.mkdtemp(prefix="hexbridge_htp_"))
    ctx_path = work / "model_ctx.onnx"

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_file_path", str(ctx_path))

    started = time.time()
    try:
        create_session(
            model_path,
            {
                "htp_arch": htp_arch,
                "soc_model": soc_model,
                "htp_graph_finalization_optimization_mode": "3",
                "offload_graph_io_quantization": "1",
            },
            so,
        )
    except Exception as e:
        result.update(ok=False, error=str(e).splitlines()[0][:300],
                      compile_s=round(time.time() - started, 1))
        shutil.rmtree(work, ignore_errors=True)
        return result
    result["compile_s"] = round(time.time() - started, 1)

    if not ctx_path.exists():
        result.update(ok=False, error="Compiler produced no context model — nothing was claimed by QNN")
        shutil.rmtree(work, ignore_errors=True)
        return result

    ops = Counter(n.op_type for n in onnx.load(str(ctx_path), load_external_data=False).graph.node)
    shutil.rmtree(work, ignore_errors=True)

    cpu_ops = {op: n for op, n in ops.items() if op != "EPContext" and op not in _BOUNDARY_OPS}
    result.update(
        npu_graphs=ops.get("EPContext", 0),
        cpu_ops=cpu_ops,
        boundary_ops=sum(ops.get(op, 0) for op in _BOUNDARY_OPS),
        ok=ops.get("EPContext", 0) == 1 and not cpu_ops,
    )
    return result
