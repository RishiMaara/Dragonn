# Evidence

Every number in the [project README](../../README.md) traces to a file in this
folder. The README's Qualcomm AI Hub job links only open for the account owner;
these files hold the same results and anyone can read them.

All device results come from **Snapdragon X Elite CRD** on Qualcomm AI Hub
(SC8380XP, Hexagon v73, QNN SDK 2.45, ONNX Runtime 1.27.1).

## Real hardware

| README claim | File | AI Hub jobs |
|---|---|---|
| Shipped model on the NPU: 185/187 layers, **17.8 ms** median, 43 MB, 5.2 s cold / 0.5 s warm load; HTP output cosine 0.9992 vs CPU, 0.9925 vs FP32 | [`aihub_validation.json`](aihub_validation.json) | profile `jpxlee83p`, inference `jg9z9918p` |
| Same device, session 2: FP32-CPU 1009 ms, **silent fallback 1409 ms (0.72x)**, NPU 17.8 ms (57x) | [`aihub_baseline_session2.json`](aihub_baseline_session2.json) | `jp2rjx74g`, `jpyonz475`, `jpxlee83p` |
| Same device, session 1: FP32-CPU 986 ms, **silent fallback 1241 ms (0.79x)**, NPU 41.7 ms (24x) | [`aihub_baseline_session1.json`](aihub_baseline_session1.json) | `jgk2yj9ng`, `j5ql2jmop`, `jp2rm1o6g` |
| **Trap 3:** the X Elite rejecting signed-int8 LayerNorm — 18 × `backendValidateOpConfig ... error code 3110`, 14 × `Failed to finalize QNN graph` | [`device_logs/j57edm49p_FAILED_int8-weights.log`](device_logs/j57edm49p_FAILED_int8-weights.log) | profile `j57edm49p` |
| The same device accepting the fixed model — zero rejections | [`device_logs/jpxlee83p_PASSED_shipped-model.log`](device_logs/jpxlee83p_PASSED_shipped-model.log) | profile `jpxlee83p` |
| WER with the encoder on the NPU: **14.07%** vs 12.70% on CPU | [`wer_report.json`](wer_report.json) | inference `jg9z991wp` |
| Why: per clip, the NPU's encoder outputs vs CPU (median cosine 0.9987); 11 transcripts changed — 6 worse, 2 better, net +11 errors; sign test p ≈ 0.29 | [`wer_npu_vs_cpu_per_clip.json`](wer_npu_vs_cpu_per_clip.json) | outputs of `jg9z991wp`, decoded locally |

## Local

| README claim | File | Reproduce |
|---|---|---|
| Scanner flags the rejected model (SPLIT, LayerNorm ×9) and passes the shipped one (one NPU graph) | [`scanner_whisper-tiny-int8w.json`](scanner_whisper-tiny-int8w.json), [`scanner_whisper-tiny-qdq.json`](scanner_whisper-tiny-qdq.json) | `python -m scanner --input <model> --compile-check` |
| WER and encoder cosine: real vs synthetic calibration (12.70% vs 13.33%), 16-bit vs 8-bit activations (12.70% vs 93.5%) | [`wer_calibration_and_precision.json`](wer_calibration_and_precision.json) | `python -m scripts.eval_wer --variant ...` |
| Pipeline: coverage, local HTP compile, x86 CPU baseline | [`pipeline_results.json`](pipeline_results.json), [`coverage_report.json`](coverage_report.json), [`pitch_summary.txt`](pitch_summary.txt) | `python -m scripts.run_pipeline ...` |

## Reproducing the device results

The shipped model rebuilds bit-for-bit from the pipeline, so re-running the
device jobs tests the exact model these files describe. With your own free AI
Hub account:

```bash
python -m scripts.aihub_validate
python -m scripts.aihub_validate --cpu-baseline --profile-job <your-profile-job-id>
python -m scripts.eval_wer --aihub
```
