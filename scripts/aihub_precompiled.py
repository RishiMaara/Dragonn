"""
Profile an already-compiled QNN model on a real Snapdragon device.

Qualcomm publishes its AI Hub models pre-compiled for each chipset (an
EPContext .onnx plus its QAIRT context binary). Those assets are the answer to
"what does a model that Qualcomm themselves optimised for the NPU cost?" — and
the honest baseline to measure this project's own conversions against.

Nothing here converts anything: it uploads the compiled asset, profiles it on
the device, and records what the device reported.

    python -m scripts.aihub_precompiled --dir <asset-dir> --name whisper-tiny-encoder
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

from scripts.aihub_validate import _with_network_retry, summarize_profile

logger = logging.getLogger("hexagon-bridge.aihub")

DEFAULT_DEVICE = "Snapdragon X Elite CRD"


def stage_for_hub(model_dir: Path) -> Path:
    """
    Copy a model into a layout AI Hub accepts: every companion file has to carry
    the .onnx file's basename ("ONNX model and weight file must have same
    basename"), which neither ONNX Runtime's `model.onnx.data` convention nor
    Qualcomm's own `encoder_qairt_context.bin` release assets satisfy.

    Returns the staged directory. The original is left untouched.
    """
    import onnx

    model_dir = Path(model_dir)
    model_path = model_dir if model_dir.is_file() else next(iter(sorted(model_dir.glob("*.onnx"))))
    stem = model_path.stem
    staged = Path(tempfile.mkdtemp(prefix="hexbridge_hub_"))

    model = onnx.load(str(model_path), load_external_data=False)

    # Compiled QNN graph: the EPContext node names its context binary.
    for node in model.graph.node:
        if node.op_type != "EPContext":
            continue
        for attribute in node.attribute:
            if attribute.name == "ep_cache_context" and attribute.s:
                source = model_path.parent / Path(attribute.s.decode()).name
                if source.exists():
                    shutil.copy(source, staged / f"{stem}.bin")
                    attribute.s = f"./{stem}.bin".encode()

    # External weights: every initializer records the file it lives in.
    renamed: set[str] = set()
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location":
                renamed.add(entry.value)
                entry.value = f"{stem}.data"
    for original in renamed:
        source = model_path.parent / original
        if source.exists():
            shutil.copy(source, staged / f"{stem}.data")

    onnx.save(model, str(staged / model_path.name))
    return staged


def profile_precompiled(model_dir: Path, name: str, device_name: str, options: str = "") -> dict:
    import qai_hub as hub

    device = hub.Device(device_name)
    model_dir = stage_for_hub(model_dir)
    logger.info(f"[{name}] uploading {model_dir}")
    model = _with_network_retry(
        lambda: hub.upload_model(str(model_dir)), None, f"[{name}] upload"
    )
    job = hub.submit_profile_job(model=model, device=device, name=f"hexagon-bridge {name}",
                                 options=options)
    logger.info(f"[{name}] profile job: {job.url}")

    status = _with_network_retry(lambda: job.wait(), job, f"[{name}] profile")
    if not status.success:
        return {"name": name, "job": job.job_id, "url": job.url,
                "failed": status.message or "profile job failed"}

    profile = _with_network_retry(lambda: job.download_profile(), job, f"[{name}] download")
    summary = summarize_profile(profile)
    return {"name": name, "job": job.job_id, "url": job.url, **summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, type=Path, help="Directory holding the compiled model")
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--options", default="", help='AI Hub profile options, e.g. "--onnx_execution_providers qnn --compute_unit all" for an uncompiled ONNX')
    parser.add_argument("--json", type=Path, help="Where to write the report")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s",
                        datefmt="%H:%M:%S")

    result = profile_precompiled(args.dir, args.name, args.device, args.options)
    result["device"] = args.device
    result["options"] = args.options
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(f"Wrote {args.json}")
    return 0 if not result.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
