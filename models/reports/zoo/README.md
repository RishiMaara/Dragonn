# Model zoo: the silent failure, across popular models

Each model through two paths — **naive** (`quantize_dynamic`, what most tutorials show) and **Hexagon Bridge** (static a16w8 QDQ, real calibration data) — checked by the scanner, Qualcomm's HTP compiler run locally, and accuracy on held-out real data.
Reproduce: `python -m scripts.model_zoo`.

| Model | Use on a laptop | Naive path | Hexagon Bridge | Accuracy: FP32 → naive → bridge | Size FP32 → bridge |
|---|---|---|---|---|---|
| [minilm](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Text embeddings for local semantic search / RAG | 0% NPU (dynamic quantization) | 2 NPU graphs; CPU: GatherND ×1 | search results matching FP32: 1.0 → 0.9576 → 0.987 | 86.19 → 32.93 MB |
