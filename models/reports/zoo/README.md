# Model zoo: the silent failure, across popular models

Each model through two paths — **naive** (`quantize_dynamic`, what most tutorials show) and **Hexagon Bridge** (static a16w8 QDQ, real calibration data) — checked by the scanner, Qualcomm's HTP compiler run locally, and accuracy on held-out real data.
Reproduce: `python -m scripts.model_zoo`.

| Model | Use on a laptop | Naive path | Hexagon Bridge | Accuracy: FP32 → naive → bridge | Size FP32 → bridge |
|---|---|---|---|---|---|
| [minilm](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Text embeddings for local semantic search / RAG | wrong format (dynamic quantization) — 1 NPU graph(s); CPU: Gather ×1 | one NPU graph | same top search result: 100.0 → 63.3 → 75.0 | 86.15 → 33.03 MB |
| [distilbert-sst2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | On-device text classification (sentiment) | wrong format (dynamic quantization) — 1 NPU graph(s); CPU: Gather ×1 | one NPU graph | SST-2 accuracy: 90.7 → 90.7 → 91.0 | 255.48 → 86.44 MB |
| [mobilenetv2](https://huggingface.co/google/mobilenet_v2_1.0_224) | Image classification | wrong format (dynamic quantization) — one NPU graph | one NPU graph | Imagenette top-1 accuracy: 79.5 → 7.2 → 79.5 | 13.67 → 3.96 MB |
| [clip-vision](https://huggingface.co/openai/clip-vit-base-patch32) | Image embeddings for photo search (zero-shot) | wrong format (dynamic quantization) — one NPU graph | one NPU graph | zero-shot accuracy: 98.2 → 98.8 → 98.2 | 335.24 → 84.59 MB |
| [whisper-base](https://huggingface.co/openai/whisper-base) | Speech recognition (encoder) | wrong format (dynamic quantization) — 2 NPU graph(s); CPU: DynamicQuantizeLinear ×2, ConvInteger ×2 | one NPU graph | word error rate: 8.84 → 11.33 → 8.84 | 78.6 → 20.64 MB |

## On a real Snapdragon X Elite

Written by `python -m scripts.aihub_precompiled` — Qualcomm AI Hub, device `Snapdragon X Elite CRD`. Local checks predict; only this settles it.

| Model | Median | Placement | Outcome |
|---|---|---|---|
| clip-vision-bridge | 2.55 ms | 2 on CPU, 482 on NPU | ✅ ran |
| clip-vision-naive | 3.06 ms | 450 on NPU | ✅ ran |
| distilbert-sst2-bridge | 2.44 ms | 248 on NPU, 1 on CPU | ✅ ran |
| minilm-bridge | 1.29 ms | 245 on NPU, 1 on CPU | ✅ ran |
| minilm-naive | 1.09 ms | 2 on CPU, 232 on NPU | ✅ ran |
| mobilenetv2-bridge | 0.36 ms | 2 on CPU, 69 on NPU | ✅ ran |
| whisper-base-bridge | 45.7 ms | 2 on CPU, 241 on NPU | ✅ ran |
| whisper-base-naive | — | — | ❌ The process ended because of an access violation. Consult the runtime log for more details. |
