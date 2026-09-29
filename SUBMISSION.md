# Hexagon Bridge — Snapdragon® AI Lab Build & Present Challenge 2026

**Make the Hexagon NPU reachable for models that aren't in anyone's catalog — and
prove, on the laptop itself, that the NPU is really the thing running them.**

Author: Rishi (RishiMaara) · [github.com/RishiMaara/Dragonn](https://github.com/RishiMaara/Dragonn) · MIT licensed

---

## The problem

A Snapdragon X Elite HP laptop ships with a 45 TOPS Hexagon NPU. Take any model
that isn't already in a vendor catalog — a fine-tune, a newer checkpoint — and
getting it onto that NPU fails in ways that produce **no error message**:

- The standard quantization recipe (`quantize_dynamic`) yields a model that runs
  **0% on the NPU** — and **26–28% slower than not quantizing at all**, measured
  twice on a real X Elite.
- ONNX Runtime's documented provider argument silently gives you a **CPU-only
  session**, while your code logs "NPU".
- A model can pass every local check and still be **rejected by the chip** —
  once over one sign bit on one input of one op type.

Every one of those failures is *silent*. The app works. The battery doesn't.

## What I built

A pre-flight check, a conversion pipeline, and a real-hardware validation loop.

| | |
|---|---|
| **Scanner** | Answers "will this model run on the Hexagon NPU, and if not, which ops and why" on any Windows PC — no Snapdragon device needed. Static op/precision rules, then Qualcomm's own HTP compiler run locally (~4 s). |
| **Converter** | Static a16w8 QDQ quantization done the QNN-specific way, calibrated on real speech. |
| **Validator** | Runs the model on a real Snapdragon X Elite through Qualcomm AI Hub and reconciles the prediction against what the device did. |
| **App** | Speech-to-text server (OpenAI-compatible) + dashboard that reports the execution provider that actually attached — never the one requested. |

It composes the existing solutions rather than inventing new ones: ONNX Runtime's
strict no-CPU-fallback session, Olive's attention-mask surgery, Qualcomm's own
mask convention and AI Hub models. The gap it fills is **diagnosis** — ORT's
strict mode says yes/no, AI Hub needs a cloud upload per model, Olive applies a
fix only once you know which one you need.

## Verified on real Snapdragon hardware

Device: `Snapdragon X Elite CRD` (SC8380XP, Hexagon v73) via Qualcomm AI Hub.
Every number below is committed as machine-readable evidence in
[`models/reports/`](models/reports/README.md), indexed claim by claim.

| Claim | Number | Evidence |
|---|---|---|
| Encoder runs on the NPU | 185 / 187 layers, 98.4% of device time | `aihub_validation.json` |
| Latency | **17.8 ms** median (100 runs) | profile `jpxlee83p` |
| The silent fallback costs speed | quantized-on-CPU is **0.79x / 0.72x** of plain FP32 on CPU, in two sessions | `aihub_baseline_session1/2.json` |
| Quantization costs no accuracy on CPU | **12.70% WER = the FP32 reference**, 57 held-out clips | `wer_calibration_and_precision.json` |
| The NPU costs a little accuracy | 14.07% WER (+1.37, sign test p ≈ 0.29) | `wer_report.json` |
| The scanner catches what the chip rejects | model rejected by the X Elite is flagged statically and by the local compiler | `device_logs/j57edm49p_FAILED_int8-weights.log` |
| Skipping quantization is not a shortcut | FP32 passes locally, fails on the device (error 6000), twice | `aihub_fp32_on_htp_FAILED.json` |
| Reproducible bit-for-bit | rerunning the pipeline rebuilds the exact validated model (SHA-256 pinned) | `pipeline_results.json` |
| A decoder *can* live on this NPU | Qualcomm's own Whisper decoder: **2.60 ms/token**, 509/509 layers on NPU; their encoder 25.0 ms vs this pipeline's 17.8 ms | `aihub_qualcomm_whisper.json` |

## It isn't one model

Five popular models, each through the naive path and this pipeline
([`models/reports/zoo/README.md`](models/reports/zoo/README.md), reproduce with
`python -m scripts.model_zoo`):

| Model | What it's for | Accuracy: FP32 → naive → this pipeline | Size |
|---|---|---|---|
| MobileNetV2 | Image classification | 79.5% → **7.2%** → **79.5%** top-1 | 13.7 → 4.0 MB |
| Whisper-base | Speech recognition | 8.84% → 11.33% → **8.84%** WER | 78.6 → 20.6 MB |
| MiniLM-L6 | Embeddings for local search | 100% → 63.3% → 75.0% same top hit | 86.2 → 33.0 MB |
| DistilBERT SST-2 | Sentiment | 90.7% → 90.7% → **91.0%** | 255.5 → 86.4 MB |
| CLIP ViT-B/32 vision | Image search | 98.2% → 98.8% → 98.2% zero-shot | 335.2 → 84.6 MB |

All five were then profiled on the real X Elite, and all five run on the NPU:
MobileNetV2 0.36 ms, MiniLM 1.29 ms, DistilBERT 2.44 ms, CLIP 2.55 ms,
whisper-base 45.7 ms.

MobileNetV2 is the clearest case of the trap this project exists for: the naive
model passes every *placement* check — one NPU graph, nothing on CPU — and
classifies 7% of images correctly. It needs per-channel weights, which nothing in
the toolchain tells you. And the naive whisper-base didn't just lose accuracy: it
**crashed the device runtime** with an access violation.

## Why this matters on a Snapdragon-powered HP PC

The NPU is the reason to buy the laptop. Today it is reachable only for models
someone else already converted. This turns "we should use the NPU" into a
checkable claim for **your** model, on **your** machine, in seconds — and makes
the failure modes loud instead of silent.

## Try it on an HP OmniBook

```powershell
git clone https://github.com/RishiMaara/Dragonn; cd Dragonn
powershell -ExecutionPolicy Bypass -File .\setup-snapdragon.ps1
```

The setup script refuses to pass quietly: it checks for ARM64-native Python,
installs the device dependencies, confirms ONNX Runtime can see the Hexagon NPU
(and names the driver version if it can't), then builds a session with CPU
fallback **disabled** — so "runs on the NPU" is proven on that laptop.

Then `python -m server.app` and open http://127.0.0.1:8000.

## Demo (5 minutes)

1. **The model the chip rejects** — scan it before owning the chip:
   `python -m scanner --input models/whisper-tiny-int8w/ --compile-check`
   → flagged twice, statically and by Qualcomm's compiler: 9 LayerNorms left on
   CPU, the graph split into 9 NPU fragments. No other tool in the stack complains.
2. **The fix** — scan the shipped model: 100%, one NPU graph, in about 4 seconds.
3. **The chip's verdict** — the committed device logs, side by side: this tool's
   prediction, and the X Elite's own log rejecting that first model
   (`error 3110`, nine times) and accepting the second.
4. **The app** — dashboard transcribes a held-out clip and reports which provider
   actually ran the encoder, with word error rate against the reference.
5. **Breadth** — `models/reports/zoo/README.md`: the same two paths across five
   popular models.

## Judging criteria

| Criterion | Where it shows up |
|---|---|
| **Technical implementation** | Three independent verdicts on NPU placement (static rules, Qualcomm's HTP compiler, ONNX Runtime strict mode), reconciled against a real device; a16w8 quantization with real-speech calibration; compiled-graph caching; 31 tests, each pinning a bug that once produced a wrong number; CI on Windows with the QNN plugin |
| **Use case & innovation** | On-device speech-to-text that stays on the laptop, plus the diagnostic layer nobody ships: *why* a model misses the NPU, and which known fix applies |
| **Deployment & accessibility** | One PowerShell command on a Snapdragon laptop; no Snapdragon hardware needed for the pre-flight check; MIT licensed; every claim reproducible from committed evidence |
| **Presentation & documentation** | README with each failure reproduced and measured; evidence index; honest limits stated in the same document as the wins |

## What I did not prove

- The **decoder** runs on CPU today; only the encoder is on the NPU. The speedups
  above are encoder speedups.
- No **power or battery** measurement — Qualcomm AI Hub doesn't expose it.
- Speedup is 24–57x by AI Hub's CPU baseline, but only **4–10x** against a
  well-tuned CPU; the honest figure needs a physical X Elite.
- The NPU's +1.37 WER is not statistically significant at 57 clips.
- The export and calibration path is built and proven for Whisper; other models
  need their own calibration data.
