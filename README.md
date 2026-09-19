# 🐉 Project Dragonn — "Hexagon Bridge"

> **Know whether your model will actually touch the Hexagon NPU — before you ship it, not after the battery dies.**

Hexagon Bridge is a pre-flight check, conversion pipeline and real-hardware
validation harness for running models on the Hexagon NPU in Snapdragon X PCs. It
exports, quantizes the *QNN-specific* way, and — the part nobody else does — tells
you the truth about what will run on the NPU and what will silently fall back to CPU.

**Scope, precisely:** the scanner, the local HTP compile check and the AI Hub
validation work on any ONNX model. The export and calibration path is built and
proven for **Whisper** (whisper-tiny's encoder), end to end, on a real Snapdragon X Elite.

## Results at a glance

Measured on a real **Snapdragon X Elite** (Qualcomm AI Hub, QNN SDK 2.45) — every
number links to its job in [Results](#results):

| | |
|---|---|
| **Runs on the NPU** | 185 of 187 layers on the Hexagon NPU; the other 2 are the float↔int conversions at the graph edges, on CPU by design |
| **Speed** | **17.8 ms** median encoder latency — **57x** faster than FP32 on the same device's CPU by AI Hub's numbers; **4–10x** against a well-tuned CPU (see caveats) |
| **The silent fallback** | A quantized model that misses the NPU runs **26–28% slower than not quantizing at all** — measured in two separate sessions |
| **Accuracy** | Quantized encoder on CPU: **12.70% WER = the FP32 reference**. On the NPU: 14.07% (+1.37, not statistically significant at this sample size) |
| **Catches the failures** | The scanner flags the exact model a real X Elite rejected — statically, and by running Qualcomm's HTP compiler locally in ~4 s |
| **Cold start** | Compiled NPU graph cached: session start 3.3 s → 0.1 s (device: 5.2 s cold vs 0.5 s warm) |

---

## The Problem

A Snapdragon X Elite / X Plus laptop ships with a Hexagon NPU rated at 45 TOPS
(80 on X2 Elite). For any model that isn't already sitting in a vendor catalog,
that NPU is effectively unreachable — not because the silicon can't run the model,
but because the path from *"checkpoint on HuggingFace"* to *"graph the Hexagon
NPU will accept"* is a specialist pipeline with **silent** failure modes.

Three things break, in order of how much they hurt:

**1. The curated catalogs only cover what's already in them.**
Qualcomm AI Hub and the Windows ML / Foundry Local model catalogs ship
pre-converted, pre-quantized models that do run on the NPU. Step one inch outside
that list — a domain fine-tune, a newer checkpoint, a model nobody at Qualcomm
prioritized — and there is no supported path. You are on your own.

**2. The local-inference stacks people actually use don't target QNN.**
llama.cpp, Ollama and LM Studio load GGUF and execute on CPU (and increasingly
Adreno GPU). Hexagon backend work exists but is recent and partial. So the default
experience on a 45 TOPS machine is: NPU at 0%, CPU pinned, fans up, battery down.

**3. The DIY path fails silently, not loudly. ← this is the one that matters**

If you quantize an ONNX model the *conventional* way and hand it to ONNX Runtime
with QNN EP enabled:

- the session **loads successfully**
- inference **returns correct outputs**
- no exception, no warning, no error code
- and **every single node runs on CPU**

The NPU was never touched. Nothing told you. And it's worse than "no speedup": on a
real X Elite, the quantized model running on CPU was **26–28% slower than the
unquantized original**, because every quantize/dequantize pair becomes CPU work.

### This project's own first output was the proof

The first version of this pipeline ran without error and reported success. What
its quantizer actually emitted:

```
DynamicQuantizeLinear ×2      ← dynamic quantization
ConvInteger           ×1      ← QNN EP has no builder for this
MatMulInteger         ×1      ← QNN EP has no builder for this
QuantizeLinear/DequantizeLinear ×0   ← no QDQ node units at all
```

QNN EP consumes **QDQ node units**; it has no support for the dynamic-quantization
op family. That model would have loaded on a Snapdragon, produced correct
transcripts, and run **100% on CPU**. Meanwhile the scanner graded it **"85.0%
NPU-eligible"**, because its registry marked `MatMulInteger` as supported. Two green
numbers, both wrong, pointing the same wrong direction.

**Conversion isn't hard. It's deceptive.**

---

## Three traps, stacked

Fixing one exposed the next. Each one passes every check before it.

### Trap 1 — the wrong format: 0% NPU, no error

Dynamic quantization (above). Fix: static QDQ via ORT's QNN config helpers.

### Trap 2 — the right format at the wrong precision: 100% NPU, wrong answers

Measured on the whisper-tiny encoder, cosine similarity against FP32 on held-out
input, identical calibration set:

| Activation precision | Cosine vs FP32 | Verdict |
|---|---|---|
| uint8 (`a8w8`) | **0.553** | structurally perfect, numerically useless |
| uint16 (`a16w8`) | **0.995** | usable |

This encoder's outputs span roughly [-18, 18]. 256 levels cannot represent that
through nine LayerNorms without destroying the signal.

### Trap 3 — only the chip knows

The a16w8 model then scanned 100% NPU-eligible, measured cosine 0.99 locally, and
**failed on a real Snapdragon X Elite**: `Failed to finalize QNN graph` (errors 6000 /
1002). The device log showed why — the HTP backend rejected all 9 LayerNorms
(`backendValidateOpConfig ... error 3110`) because their gamma was quantized as
*signed* int8. The op is supported, the format is right, the precision is right:
one sign bit on one input of one op type.

Unsigned uint8 weights compiled the entire encoder as a single NPU graph and passed
on the device. A second fix rode along: running ORT's generic preprocessing before
the QNN one left 6 bare `Erf` nodes (rejected by HTP, each splitting the graph)
instead of fusing them into a native `Gelu`.

**The scanner now catches trap 3 two ways**, both verified against the model the
chip rejected:

| Model | Static check | Local HTP compiler (`--compile-check`) |
|---|---|---|
| int8 weights — **failed on the X Elite** | 🟡 SPLIT: LayerNorm ×9 rejected (93.9%) | 🔴 Split into 9 NPU graphs, LayerNorm ×9 on CPU |
| uint8 weights — **passed on the X Elite** | 🟢 100% | 🟢 One NPU graph, 4 s |

---

## What "suitable for Snapdragon PCs" actually requires

| # | Constraint | Why | Naive path violates it? |
|---|---|---|---|
| 1 | **Static** QDQ quantization | HTP is fixed-point; it fuses `DequantizeLinear → Op → QuantizeLinear` node units | ✅ `quantize_dynamic()` emits ops QNN can't build |
| 2 | **uint16 activations**, **unsigned** uint8 weights ("a16w8") | 8-bit activations: cosine 0.553. Signed int8 gamma: LayerNorm rejected on silicon | ✅ `activation_type=QInt8`; "int8 weights" |
| 3 | **Static input shapes** | HTP compiles a fixed graph; symbolic dims can't be resolved | ✅ exporters default to dynamic axes |
| 4 | **ARM64-native Python** | `QnnHtp.dll` is ARM64; x64 Python under Prism can't load it | ✅ silently no QNN EP |
| 5 | **`onnxruntime-qnn` 2.x, attached correctly** | It's a plugin: invisible until registered, and the classic `providers=[...]` argument yields a **CPU-only session with no error** | ✅ every tutorial's `providers=[...]` |
| 6 | **Cache the compiled graph** | HTP compilation is slow: 5.2 s cold vs 0.5 s warm on X Elite | ✅ recompiles every start |

All six are handled here. Constraint 5 is its own silent failure, inside ONNX
Runtime's API: [`scripts/qnn_ep.py`](scripts/qnn_ep.py) registers the plugin,
attaches it via `add_provider_for_devices`, and raises if `get_providers()` doesn't
show QNN — no session labelled "NPU" ever runs on CPU.

---

## The Solution

```
HF model → ONNX (static) → QNN static QDQ, real-speech calibration
         → scanner (static rules + real HTP compiler) → AI Hub: real X Elite
         → transcription server (encoder on NPU) + dashboard
```

| Component | What it does |
|---|---|
| `converter/hf_to_onnx.py` | Export → ONNX with static shapes |
| `converter/quantize.py` | Static a16w8 QDQ via ORT's QNN helpers; real-speech calibration; format self-check |
| `scanner/` | Registry + per-node quantization rules + **local HTP compile check** — the deliverable |
| `scripts/qnn_ep.py` | Attaches QNN EP correctly, verifies it, caches compiled graphs |
| `scripts/aihub_validate.py` | Real X Elite: placement, accuracy on HTP, latency distribution, CPU baseline |
| `scripts/eval_wer.py` | Word error rate on held-out speech — locally and with the encoder on a real NPU |
| `scripts/transcriber.py` | Speech-to-text: encoder on NPU (ONNX), decoder on CPU (PyTorch) |
| `server/` + `dashboard/` | OpenAI-compatible API; live dashboard with upload, microphone, and scored samples |
| `tests/` | 28 tests — each pins a bug that produced a wrong number at some point |

---

## Setup

The pipeline splits across two machines, because it has to.

**Prep host (x64 or ARM64)** — export, quantize, scan, local HTP compile:

```bash
pip install -r requirements.txt
```

```bash
pip install onnxruntime-qnn
```

The second line is optional on x64: it enables the local HTP compile check
(Qualcomm's compiler, compile-only — x64 has no NPU to execute on).

**Snapdragon X device (ARM64)** — run on the NPU:

```bash
python -c "import platform; print(platform.machine())"
```

That must print `ARM64` (not `AMD64`, which means x64 Python under emulation). Then:

```bash
pip install -r requirements-device.txt
```

## Quick Start

```bash
python -m scripts.fetch_speech
```

```bash
python -m scripts.run_pipeline --model openai/whisper-tiny --quantized-dir ./models/whisper-tiny-qdq
```

```bash
python -m scanner --input ./models/whisper-tiny-qdq/ --compile-check
```

```bash
python -m server.app
```

Then open http://127.0.0.1:8000. Accuracy and tests:

```bash
python -m scripts.eval_wer
```

```bash
python -m pytest
```

## Validate on Real Hardware (no Snapdragon PC required)

No mainstream cloud rents Snapdragon X VMs with NPU access (Azure and AWS Arm
instances use Ampere, Cobalt or Graviton chips — no Hexagon). **Qualcomm AI Hub**
hosts real Snapdragon X Elite devices and runs ONNX models on them through ONNX
Runtime + QNN EP, the same path this project deploys on. It's free; create an
account at [aihub.qualcomm.com](https://aihub.qualcomm.com), then:

```bash
qai-hub configure --api_token <YOUR_TOKEN>
```

If Windows says `qai-hub` is not recognized (per-user pip installs aren't on PATH):

```powershell
& "$env:APPDATA\Python\Python314\Scripts\qai-hub.exe" configure --api_token <YOUR_TOKEN>
```

```bash
python -m scripts.aihub_validate
```

```bash
python -m scripts.aihub_validate --cpu-baseline --profile-job <npu-profile-job-id>
```

```bash
python -m scripts.eval_wer --aihub
```

Validation reports placement by layer count **and** time share, accuracy on HTP
vs CPU and vs FP32, and the full latency distribution — median, not AI Hub's
headline number, which is its fastest run. Uploads and job polling retry through
network drops; `--profile-job` / `--inference-job` re-attach to running jobs.

---

## Demo Flow (5 minutes)

1. **The trap.** Build the model a real X Elite rejected, and scan it:

   ```bash
   python -m converter.quantize --input ./models/whisper-tiny-onnx --output ./models/whisper-tiny-int8w --weight-type INT8 --audio-dir data/speech/calib --calibration-samples 16
   ```

   ```bash
   python -m scanner --input ./models/whisper-tiny-int8w/ --compile-check
   ```

   🟡 SPLIT — 9 LayerNorms rejected, 9 NPU fragments. This model *loads without
   error* and died on silicon.
2. **The fix.** Same scan on the shipped model: 🟢 one NPU graph.
3. **The receipt.** The same-device table below: the silent fallback is slower
   than doing nothing; the NPU is 57x faster by AI Hub's numbers.
4. **It's real.** Dashboard → *LibriSpeech sample*: transcript, reference, WER,
   and which provider actually ran the encoder. Then speak into the microphone.

---

## Results

Device: `Snapdragon X Elite CRD` via Qualcomm AI Hub (SC8380XP, Hexagon v73,
QNN SDK 2.45, ONNX Runtime 1.27.1). Model: whisper-tiny encoder, 8.2M params,
static `[1,80,3000]` → `[1,1500,384]`, a16w8, 31.5 MB → 8.5 MB (3.7x).

### On the NPU — shipped model

Profile [jpxlee83p](https://workbench.aihub.qualcomm.com/jobs/jpxlee83p/),
inference [jg9z9918p](https://workbench.aihub.qualcomm.com/jobs/jg9z9918p/):

| | |
|---|---|
| Placement | 185 / 187 layers on NPU (2 = graph-edge quantize/dequantize); NPU time share 98.4% |
| Scanner prediction | 100% — **confirmed** by the device |
| Latency (100 runs) | **median 17.8 ms**, p90 18.1 ms, min 17.6 ms |
| Load | 5.2 s cold (includes HTP compile) / 0.5 s warm |
| Peak memory | 43 MB |
| Accuracy on HTP | cosine **0.9992** vs same model on CPU; **0.9925** vs FP32 |

### Same device: CPU vs NPU

Measured twice, in separate sessions
(session 1: [jgk2yj9ng](https://workbench.aihub.qualcomm.com/jobs/jgk2yj9ng/),
[j5ql2jmop](https://workbench.aihub.qualcomm.com/jobs/j5ql2jmop/),
[jp2rm1o6g](https://workbench.aihub.qualcomm.com/jobs/jp2rm1o6g/) — previous calibration;
session 2: [jp2rjx74g](https://workbench.aihub.qualcomm.com/jobs/jp2rjx74g/),
[jpyonz475](https://workbench.aihub.qualcomm.com/jobs/jpyonz475/),
[jpxlee83p](https://workbench.aihub.qualcomm.com/jobs/jpxlee83p/) — shipped model):

| Configuration | Session 1 median | Session 2 median | vs FP32 on CPU |
|---|---|---|---|
| FP32 on CPU — what people run today | 986 ms | 1009 ms | 1.0x |
| Quantized on CPU — **the silent fallback** | 1241 ms | 1409 ms | **0.79x / 0.72x** |
| Quantized on NPU — this project | 41.7 ms | **17.8 ms** | **24x / 57x** |

- **The fallback penalty reproduced in both sessions** (26%, 28% slower).
- **Treat 24–57x as an upper bound.** AI Hub doesn't log its CPU thread settings,
  and ~1 s is slow for this model: the same FP32 encoder ran in 179 ms on an x86
  Ryzen 5 5600H. Against that, the NPU advantage is **4–10x**. The truth needs a
  tuned CPU run on a physical X Elite.
- **NPU latency varies between sessions**: session 1 was bimodal (median 41.7 ms,
  best 17.6 ms), with the CPU-side graph-edge conversions taking half the time;
  session 2 was tight at 17.8 ms. Same graph structure — so device conditions, not
  the model. Quote the range.

### Accuracy — word error rate on real speech

57 held-out LibriSpeech clips (325 s, 803 words), disjoint from the 16 calibration
clips ([`scripts/eval_wer.py`](scripts/eval_wer.py); NPU run
[jg9z991wp](https://workbench.aihub.qualcomm.com/jobs/jg9z991wp/)):

| Encoder | WER | vs reference |
|---|---|---|
| PyTorch reference (whisper-tiny as released) | 12.70% | — |
| ONNX FP32 export | 12.70% | +0.00 — export is exact |
| Quantized, synthetic-audio calibration | 13.33% | +0.63 |
| **Quantized, real-speech calibration (shipped)** — CPU | **12.70%** | **+0.00** |
| **Same model on the X Elite NPU** | **14.07%** | **+1.37** |

What happened on the NPU, clip by clip: 11 of 57 transcripts changed — 6 got worse,
2 better, 3 moved sideways — net 11 more errors. The NPU's encoder outputs match
CPU at median cosine 0.9987, and the clips that changed are *not* outliers
(0.9983–0.9995): it's uniform fixed-point drift tipping near-tie decoder choices
("heard the raps" → "had her wraps"). A sign test on 6-worse / 2-better gives
p ≈ 0.29 — consistent with a small real cost, not significant at this sample
size. Settling it needs a larger evaluation set.

Absolute WER is above whisper-tiny's published LibriSpeech figure because this
small subset is an Oz book dense with invented names ("Polychrome", "Ruggedo");
compare rows, not the absolute number. Cosine alone would have hidden both
effects: switching to real-speech calibration moved encoder cosine only from 0.9937
to 0.9940 while removing all 5 extra word errors, and the NPU's 0.999 agreement
with CPU still hid 11.

---

## Honest Status

**Proven:**

- ✅ Runs on a real Snapdragon X Elite: every compute layer on the NPU, confirmed by the device
- ✅ Real transcription — upload, microphone, OpenAI-compatible API — never fake text (tested)
- ✅ Traps 1 and 3 caught by the scanner (statically and via the real HTP compiler);
  trap 2 caught by the accuracy checks (cosine on device, WER on real speech)
- ✅ The silent-fallback penalty, measured twice on real hardware
- ✅ Quantization costs zero word errors on held-out speech, on CPU
- ✅ **Bit-for-bit reproducible**: rerunning the pipeline rebuilds the exact model
  validated on the X Elite (SHA-256 `7f84fa78…` / `67772ca7…` for `.onnx` / `.onnx.data`)
- ✅ Compiled-graph cache (3.3 s → 0.1 s locally), invalidated by model or SDK changes
- ✅ 28 tests; deliberately re-breaking the LayerNorm rule makes them fail

**Not proven, or known limits:**

- ⚠️ **Only the encoder runs on the NPU.** The decoder runs on CPU (PyTorch), and on
  a Snapdragon it would dominate end-to-end transcription latency — locally the
  decoder takes ~640 ms vs the encoder's ~350 ms on CPU. The speedups above are
  encoder speedups. Decoder on NPU (static KV cache) is the next step
- ⚠️ NPU WER +1.37 points vs CPU — likely small real cost; not yet significant
- ⚠️ Speedup is 24–57x by AI Hub's CPU numbers, 4–10x against a well-tuned CPU
- ⚠️ **No power or battery measurement.** AI Hub doesn't expose it; needs a physical device
- ⚠️ Local QNN (2.50) and the device (2.45) differ: local compile passing is
  necessary, not sufficient. Confirm on device
- ⚠️ Export/calibration are Whisper-specific; other models need their own calibration data
- ⚠️ English only; like all Whisper models, it can hallucinate short words on non-speech

---

## Architecture

```
Dragonn/
├── converter/     export + QNN-correct quantization
├── scanner/       registry, quantization rules, local HTP compile check — the deliverable
├── scripts/       QNN EP attach/cache, AI Hub validation, WER eval, transcriber, pipeline
├── server/        OpenAI-compatible transcription API
├── dashboard/     live UI: upload, microphone, scored samples, provider telemetry
├── tests/         28 tests, one per real bug
├── models/        generated models (gitignored) + reports/ — the evidence, committed
└── data/          downloaded speech (gitignored; python -m scripts.fetch_speech)
```

Built for the Snapdragon AI Lab Challenge.
