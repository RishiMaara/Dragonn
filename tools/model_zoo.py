"""
Dragonn — Model Zoo: the silent failure, across popular models
=====================================================================
Five models people run on laptops, each through two quantization paths:

  naive   onnxruntime.quantization.quantize_dynamic — what most tutorials show
  bridge  Dragonn: static a16w8 QDQ (QNN helpers) with real calibration data

and three checks per path:

  scanner       static analysis (scanner/)
  HTP compiler  Qualcomm's real Hexagon compiler, run locally (scanner/htp_compile.py)
  accuracy      against the FP32 model, on held-out real data, with the metric that
                matches each model's job (search agreement, accuracy, WER)

Usage:
    python -m tools.model_zoo                    # every model
    python -m tools.model_zoo --model minilm     # one model
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger("dragonn.zoo")

ZOO_DIR = Path("models/zoo")
DATA_DIR = Path("data/zoo")
REPORT_DIR = Path("models/reports/zoo")

SEQ_LEN = 128                 # fixed text length: the HTP compiles static shapes
N_TEXT_CALIB, N_TEXT_EVAL = 64, 300
N_IMG_CALIB = 32

# Imagenette: 10 ImageNet classes. Index in ImageNet-1k, and a CLIP prompt name.
IMAGENETTE_IMAGENET_IDX = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
IMAGENETTE_NAMES = ["tench", "English springer", "cassette player", "chain saw", "church",
                    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"]


# ─────────────────────────────────────────────────────────────────────────────
# Held-out real data (calibration and evaluation splits are disjoint)
# ─────────────────────────────────────────────────────────────────────────────

def text_data() -> dict:
    """SST-2 movie-review sentences: calibrate on train, evaluate on validation."""
    cache = DATA_DIR / "sst2.json"
    if not cache.exists():
        from datasets import load_dataset
        ds = load_dataset("stanfordnlp/sst2")
        data = {
            "calib": [r["sentence"] for r in ds["train"].select(range(N_TEXT_CALIB))],
            "eval": [[r["sentence"], r["label"]] for r in ds["validation"].select(range(N_TEXT_EVAL))],
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
    return json.loads(cache.read_text(encoding="utf-8"))


def image_data() -> dict:
    """
    Imagenette photos, sampled through the Hugging Face rows API — ~200 images
    instead of a 350 MB download. Every 6th image calibrates; the rest evaluate.
    """
    folder = DATA_DIR / "imagenette"
    manifest = folder / "manifest.json"
    if not manifest.exists():
        folder.mkdir(parents=True, exist_ok=True)
        base = ("https://datasets-server.huggingface.co/rows?dataset=johnowhitaker/imagenette2-320"
                "&config=default&split=train")
        rows = []
        for offset in np.linspace(0, 13394 - 10, 20).astype(int):
            with urllib.request.urlopen(f"{base}&offset={offset}&length=10", timeout=60) as r:
                rows += json.load(r)["rows"]
        entries = []
        for i, row in enumerate(rows):
            path = folder / f"{i:03d}.jpg"
            if not path.exists():
                with urllib.request.urlopen(row["row"]["image"]["src"], timeout=60) as r:
                    path.write_bytes(r.read())
            entries.append({"file": path.name, "label": row["row"]["label"]})
        manifest.write_text(json.dumps(entries), encoding="utf-8")

    entries = json.loads(manifest.read_text(encoding="utf-8"))
    calib = [e for i, e in enumerate(entries) if i % 6 == 0][:N_IMG_CALIB]
    evals = [e for i, e in enumerate(entries) if i % 6 != 0]
    return {"folder": str(folder), "calib": calib, "eval": evals}


def speech_data() -> dict:
    root = Path("data/speech")
    if not (root / "eval" / "transcripts.jsonl").exists():
        raise SystemExit("No speech data. Run: python -m tools.fetch_speech")
    load = lambda split: [json.loads(l) for l in (root / split / "transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"calib": load("calib"), "eval": load("eval"), "root": str(root)}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _export(module, example_inputs: tuple, input_names: list, output_names: list, path: Path):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False keeps the TorchScript exporter. PyTorch's newer dynamo
    # exporter decomposes scaled-dot-product attention into a graph carrying
    # GatherND and IsNaN, which the HTP compiler leaves on CPU — splitting the
    # graph in two and, on a real X Elite, failing to allocate memory at all.
    torch.onnx.export(module.eval(), example_inputs, str(path), input_names=input_names,
                      output_names=output_names, opset_version=17, dynamo=False)


def _session(path: Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def _run_all(path: Path, feeds: list) -> list:
    s = _session(path)
    return [s.run(None, f)[0] for f in feeds]


def _cos(a, b) -> float:
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _text_feeds(tokenizer, sentences: list) -> list:
    enc = tokenizer(sentences, padding="max_length", max_length=SEQ_LEN, truncation=True, return_tensors="np")
    return [{"input_ids": enc["input_ids"][i:i + 1].astype(np.int64),
             "attention_mask": enc["attention_mask"][i:i + 1].astype(np.int64)} for i in range(len(sentences))]


def _image_feeds(processor, folder: str, entries: list) -> list:
    from PIL import Image
    feeds = []
    for e in entries:
        img = Image.open(Path(folder) / e["file"]).convert("RGB")
        feeds.append({"pixel_values": processor(images=img, return_tensors="np")["pixel_values"].astype(np.float32)})
    return feeds


# ─────────────────────────────────────────────────────────────────────────────
# The five models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ZooModel:
    name: str
    hf_id: str
    use_case: str
    metric_name: str
    headline_key: str
    export: Callable      # (out_path) -> None
    calib_feeds: Callable  # () -> list[feeds]
    evaluate: Callable     # ({variant: path}) -> {variant: {metric...}}
    per_channel: bool = False   # depthwise convolutions need per-channel weights


def _minilm() -> ZooModel:
    import torch
    from transformers import AutoModel, AutoTokenizer
    hf = "sentence-transformers/all-MiniLM-L6-v2"
    tok = AutoTokenizer.from_pretrained(hf)

    class Wrap(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, input_ids, attention_mask):
            return s.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def export(out):
        f = _text_feeds(tok, ["example"])[0]
        _export(Wrap(AutoModel.from_pretrained(hf, attn_implementation="eager")),
                (torch.from_numpy(f["input_ids"]), torch.from_numpy(f["attention_mask"])),
                ["input_ids", "attention_mask"], ["last_hidden_state"], out)

    def evaluate(paths):
        sentences = [s for s, _ in text_data()["eval"]]
        feeds = _text_feeds(tok, sentences)

        def embed(path):
            out = _run_all(path, feeds)
            e = np.stack([(h[0] * f["attention_mask"][0][:, None]).sum(0) / f["attention_mask"][0].sum()
                          for h, f in zip(out, feeds)])
            return e / np.linalg.norm(e, axis=1, keepdims=True)

        ref = embed(paths["fp32"])
        ref_nn = np.argmax(ref @ ref.T - 2 * np.eye(len(ref)), axis=1)
        results = {}
        for variant, path in paths.items():
            e = ref if variant == "fp32" else embed(path)
            nn = np.argmax(e @ e.T - 2 * np.eye(len(e)), axis=1)
            results[variant] = {
                "embedding_cosine_vs_fp32": round(float(np.mean(np.sum(e * ref, axis=1))), 4),
                "same_nearest_neighbour_pct": round(float(np.mean(nn == ref_nn)) * 100, 1),
            }
        return results

    return ZooModel("minilm", hf, "Text embeddings for local semantic search / RAG",
                    "same top search result", "same_nearest_neighbour_pct", export,
                    lambda: _text_feeds(tok, text_data()["calib"]), evaluate,
                    # Per-tensor weights cost this model 25 points of retrieval accuracy
                    # (75.0% -> 97.3% same top hit, cosine 0.987 -> 0.9992). More or
                    # better calibration data changed nothing; the weight granularity
                    # was the whole story.
                    per_channel=True)


def _distilbert() -> ZooModel:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    hf = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    tok = AutoTokenizer.from_pretrained(hf)

    class Wrap(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, input_ids, attention_mask):
            return s.m(input_ids=input_ids, attention_mask=attention_mask).logits

    def export(out):
        f = _text_feeds(tok, ["example"])[0]
        _export(Wrap(AutoModelForSequenceClassification.from_pretrained(hf, attn_implementation="eager")),
                (torch.from_numpy(f["input_ids"]), torch.from_numpy(f["attention_mask"])),
                ["input_ids", "attention_mask"], ["logits"], out)

    def evaluate(paths):
        data = text_data()["eval"]
        feeds = _text_feeds(tok, [s for s, _ in data])
        labels = np.array([l for _, l in data])
        preds = {v: np.array([int(np.argmax(o)) for o in _run_all(p, feeds)]) for v, p in paths.items()}
        return {v: {"accuracy_pct": round(float(np.mean(p == labels)) * 100, 1),
                    "agreement_with_fp32_pct": round(float(np.mean(p == preds["fp32"])) * 100, 1)}
                for v, p in preds.items()}

    return ZooModel("distilbert-sst2", hf, "On-device text classification (sentiment)",
                    "SST-2 accuracy", "accuracy_pct", export,
                    lambda: _text_feeds(tok, text_data()["calib"]), evaluate)


def _mobilenet() -> ZooModel:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    hf = "google/mobilenet_v2_1.0_224"
    proc = AutoImageProcessor.from_pretrained(hf)

    class Wrap(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, pixel_values): return s.m(pixel_values=pixel_values).logits

    def export(out):
        _export(Wrap(AutoModelForImageClassification.from_pretrained(hf)),
                (torch.zeros(1, 3, 224, 224),), ["pixel_values"], ["logits"], out)

    def evaluate(paths):
        d = image_data()
        feeds = _image_feeds(proc, d["folder"], d["eval"])
        truth = np.array([IMAGENETTE_IMAGENET_IDX[e["label"]] for e in d["eval"]])
        results, ref = {}, None
        for v, p in paths.items():
            logits = np.concatenate(_run_all(p, feeds))
            offset = 1 if logits.shape[1] == 1001 else 0      # this checkpoint prepends "background"
            pred = logits.argmax(1) - offset
            ref = pred if v == "fp32" else ref
            results[v] = {"top1_accuracy_pct": round(float(np.mean(pred == truth)) * 100, 1),
                          "agreement_with_fp32_pct": round(float(np.mean(pred == ref)) * 100, 1)}
        return results

    return ZooModel("mobilenetv2", hf, "Image classification", "Imagenette top-1 accuracy",
                    "top1_accuracy_pct", export,
                    lambda: _image_feeds(proc, image_data()["folder"], image_data()["calib"]), evaluate,
                    per_channel=True)   # depthwise convs: per-tensor weights cost 78 points of top-1


def _clip_vision() -> ZooModel:
    import torch
    from transformers import CLIPModel, CLIPProcessor, CLIPVisionModelWithProjection
    hf = "openai/clip-vit-base-patch32"
    proc = CLIPProcessor.from_pretrained(hf)

    class Wrap(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, pixel_values): return s.m(pixel_values=pixel_values).image_embeds

    def export(out):
        _export(Wrap(CLIPVisionModelWithProjection.from_pretrained(hf, attn_implementation="eager")),
                (torch.zeros(1, 3, 224, 224),), ["pixel_values"], ["image_embeds"], out)

    def evaluate(paths):
        d = image_data()
        feeds = _image_feeds(proc.image_processor, d["folder"], d["eval"])
        truth = np.array([e["label"] for e in d["eval"]])
        # Zero-shot "photo search": text side stays FP32, only the image encoder varies.
        clip = CLIPModel.from_pretrained(hf).eval()
        with torch.no_grad():
            t = proc.tokenizer([f"a photo of a {n}" for n in IMAGENETTE_NAMES], padding=True, return_tensors="pt")
            text = clip.get_text_features(**t)
            text = getattr(text, "pooler_output", text)
            text = (text / text.norm(dim=-1, keepdim=True)).numpy()
        results, ref = {}, None
        for v, p in paths.items():
            img = np.concatenate(_run_all(p, feeds))
            img = img / np.linalg.norm(img, axis=1, keepdims=True)
            ref = img if v == "fp32" else ref
            results[v] = {"zero_shot_accuracy_pct": round(float(np.mean((img @ text.T).argmax(1) == truth)) * 100, 1),
                          "embedding_cosine_vs_fp32": round(float(np.mean(np.sum(img * ref, axis=1))), 4)}
        return results

    return ZooModel("clip-vision", hf, "Image embeddings for photo search (zero-shot)",
                    "zero-shot accuracy", "zero_shot_accuracy_pct", export,
                    lambda: _image_feeds(proc.image_processor, image_data()["folder"], image_data()["calib"]),
                    evaluate)


def _whisper_base() -> ZooModel:
    from speech.transcriber import WhisperTranscriber, load_audio, word_error_rate
    hf = "openai/whisper-base"

    def export(out):
        from converter.hf_to_onnx import export_whisper_to_onnx
        export_whisper_to_onnx(hf, out.parent)
        (out.parent / "encoder_model.onnx").rename(out)
        data = out.parent / "encoder_model.onnx.data"
        if data.exists():   # keep the weight file's name in step with the model's own reference
            import onnx
            m = onnx.load(str(out))
            onnx.save(m, str(out), save_as_external_data=True, all_tensors_to_one_file=True,
                      location="model.onnx.data")
            data.unlink()

    def calib():
        d, ref = speech_data(), WhisperTranscriber(None, model_id=hf)
        return [{"input_features": ref.features(load_audio((Path(d["root"]) / "calib" / r["file"]).read_bytes()))}
                for r in d["calib"]]

    def evaluate(paths):
        d = speech_data()
        results = {}
        for v, p in paths.items():
            t = WhisperTranscriber(p, model_id=hf, use_npu=False)
            refs, hyps = [], []
            for r in d["eval"]:
                out = t.transcribe(load_audio((Path(d["root"]) / "eval" / r["file"]).read_bytes()))
                refs.append(t.normalize(r["text"])); hyps.append(t.normalize(out["text"]))
            results[v] = {"wer_pct": word_error_rate(refs, hyps)["wer_percent"]}
        return results

    return ZooModel("whisper-base", hf, "Speech recognition (encoder)", "word error rate",
                    "wer_pct", export, calib, evaluate)


MODELS = {"minilm": _minilm, "distilbert-sst2": _distilbert, "mobilenetv2": _mobilenet,
          "clip-vision": _clip_vision, "whisper-base": _whisper_base}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _checks(path: Path) -> dict:
    from scanner.graph_analyzer import analyze_onnx_model
    from scanner.htp_compile import compile_check
    r = analyze_onnx_model(path)
    htp = compile_check(path)
    return {
        "scanner_npu_pct": r.effective_coverage_percent,
        "scanner_format_error": (r.format_error or {}).get("error"),
        "scanner_fallback_ops": {f["op_type"]: f["count"] for f in r.fallback_ops},
        "htp_ok": htp.get("ok"),
        "htp_npu_graphs": htp.get("npu_graphs"),
        "htp_cpu_ops": htp.get("cpu_ops"),
        "htp_error": htp.get("error"),
        "htp_compile_s": htp.get("compile_s"),
    }


def run_model(name: str) -> dict:
    from converter.quantize import model_size_mb, quantize_qnn
    from onnxruntime.quantization import quantize_dynamic

    spec = MODELS[name]()
    folder = ZOO_DIR / name
    paths = {v: folder / v / "model.onnx" for v in ("fp32", "naive", "bridge")}
    started = time.time()

    if not paths["fp32"].exists():
        logger.info(f"[{name}] exporting {spec.hf_id}")
        spec.export(paths["fp32"])
    notes = {}
    if not paths["naive"].exists():
        logger.info(f"[{name}] naive path: quantize_dynamic")
        paths["naive"].parent.mkdir(parents=True, exist_ok=True)
        try:
            quantize_dynamic(str(paths["fp32"]), str(paths["naive"]))
        except Exception as e:
            # ORT's documented remedy. Without it, quantize_dynamic crashes on
            # some torch-exported graphs (DistilBERT: "Inferred shape and
            # existing shape differ"), so give the naive path its best shot.
            notes["naive_needed_quant_pre_process"] = str(e).splitlines()[0][:160]
            logger.info(f"[{name}] quantize_dynamic failed; retrying after quant_pre_process")
            from onnxruntime.quantization.shape_inference import quant_pre_process
            work = Path(tempfile.mkdtemp(prefix="dragonn_zoo_"))
            try:
                pre = work / "pre.onnx"
                quant_pre_process(str(paths["fp32"]), str(pre), auto_merge=True)
                quantize_dynamic(str(pre), str(paths["naive"]))
            finally:
                shutil.rmtree(work, ignore_errors=True)
    if not paths["bridge"].exists():
        logger.info(f"[{name}] Dragonn path: static a16w8 QDQ, real calibration data")
        result = quantize_qnn(paths["fp32"], paths["bridge"], spec.calib_feeds(),
                              per_channel=spec.per_channel)
        notes["mask_constants_clamped"] = result["mask_constants_clamped"]
        notes["per_channel"] = result["per_channel"]

    logger.info(f"[{name}] scanner + local HTP compiler")
    checks = {}
    for variant in ("naive", "bridge"):
        try:
            checks[variant] = _checks(paths[variant])
        except Exception as e:
            checks[variant] = {"error": str(e).splitlines()[0][:200]}
            logger.error(f"[{name}] checks failed for {variant}: {checks[variant]['error']}")

    logger.info(f"[{name}] accuracy on held-out data")
    try:
        accuracy = spec.evaluate(paths)
    except Exception as e:
        accuracy = {"error": str(e).splitlines()[0][:200]}
        logger.error(f"[{name}] accuracy failed: {accuracy['error']}")

    report = {
        "model": name, "hf_id": spec.hf_id, "use_case": spec.use_case, "metric": spec.metric_name,
        "size_mb": {v: round(model_size_mb(p), 2) for v, p in paths.items()},
        "headline_key": spec.headline_key,
        "checks": checks, "accuracy": accuracy, "notes": notes,
        "wall_time_s": round(time.time() - started, 1),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _verdict(c: dict) -> str:
    wrong_format = "wrong format (dynamic quantization) — " if c["scanner_format_error"] else ""
    if c["htp_ok"]:
        return f"{wrong_format}one NPU graph" if wrong_format else "one NPU graph"
    if c["htp_error"]:
        return f"{wrong_format}compile failed"
    cpu = ", ".join(f"{op} ×{n}" for op, n in (c["htp_cpu_ops"] or {}).items())
    return f"{wrong_format}{c['htp_npu_graphs']} NPU graph(s); CPU: {cpu}"


def _headline(acc: dict, key: str) -> str:
    return f"{acc['fp32'][key]} → {acc['naive'][key]} → {acc['bridge'][key]}"


def write_summary():
    # Only this script's own per-model reports — device profiles written by
    # validate/precompiled.py live in the same folder.
    paths = [REPORT_DIR / f"{name}.json" for name in MODELS]
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths if p.exists()]
    lines = [
        "# Model zoo: the silent failure, across popular models",
        "",
        "Each model through two paths — **naive** (`quantize_dynamic`, what most tutorials "
        "show) and **Dragonn** (static a16w8 QDQ, real calibration data) — checked by "
        "the scanner, Qualcomm's HTP compiler run locally, and accuracy on held-out real data.",
        "Reproduce: `python -m tools.model_zoo`.",
        "",
        "| Model | Use on a laptop | Naive path | Dragonn | Accuracy: FP32 → naive → bridge | Size FP32 → bridge |",
        "|---|---|---|---|---|---|",
    ]
    for r in reports:
        a, s = r["accuracy"], r["size_mb"]
        if "error" in a or any("error" in c for c in r["checks"].values()):
            lines.append(f"| {r['model']} | {r['use_case']} | see {r['model']}.json | | | |")
            continue
        lines.append(
            f"| [{r['model']}](https://huggingface.co/{r['hf_id']}) | {r['use_case']} | "
            f"{_verdict(r['checks']['naive'])} | {_verdict(r['checks']['bridge'])} | "
            f"{r['metric']}: {_headline(a, r['headline_key'])} | {s['fp32']} → {s['bridge']} MB |"
        )
    device = sorted(REPORT_DIR.glob("aihub_*.json"))
    if device:
        lines += [
            "",
            "## On a real Snapdragon X Elite",
            "",
            "Written by `python -m validate.precompiled` — Qualcomm AI Hub, device "
            "`Snapdragon X Elite CRD`. Local checks predict; only this settles it.",
            "",
            "| Model | Median | Placement | Outcome |",
            "|---|---|---|---|",
        ]
        for path in device:
            r = json.loads(path.read_text(encoding="utf-8"))
            name = r.get("name", path.stem)
            if r.get("failed"):
                lines.append(f"| {name} | — | — | ❌ {r['failed'].replace('Failed to profile the model: ', '')} |")
                continue
            units = r.get("layers_by_unit") or r.get("compute_units") or {}
            placement = ", ".join(f"{n} on {u}" for u, n in units.items()) or f"{r.get('npu_time_percent')}% NPU time"
            lines.append(f"| {name} | {r.get('inference_ms_median')} ms | {placement} | ✅ ran |")

    (REPORT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run popular models through naive vs Dragonn quantization")
    parser.add_argument("--model", choices=list(MODELS), action="append",
                        help="Model(s) to run (default: all)")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "transformers", "huggingface_hub", "dragonn.scanner",
                  "dragonn.quantize", "dragonn.export", "root"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    for name in args.model or list(MODELS):
        try:
            r = run_model(name)
        except Exception as e:
            logger.error(f"[{name}] FAILED: {str(e).splitlines()[0][:200]}")
            (REPORT_DIR / f"{name}.failed.txt").write_text(str(e), encoding="utf-8")
            continue
        logger.info(f"[{name}] done in {r['wall_time_s']} s — naive: {_verdict(r['checks']['naive'])} | "
                    f"bridge: {_verdict(r['checks']['bridge'])} | {r['metric']}: "
                    f"{_headline(r['accuracy'], r['headline_key'])}")
    write_summary()


if __name__ == "__main__":
    main()
