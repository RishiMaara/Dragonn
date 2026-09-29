# Demo recording script — 5 minutes

Everything here runs on an ordinary x64 Windows PC. No Snapdragon device is
needed to record it: that is the point of the tool, and worth saying out loud.

Set up before recording: `pip install -r requirements.txt`, then
`pip install onnxruntime-qnn` (the local HTP compiler), and
`python -m tools.fetch_speech` for the audio. Terminal at a large font.

---

## 0:00 — The claim nobody checks (30s)

> "This laptop has a 45 TOPS NPU. Every AI app says it uses it. Almost none of
> them can prove it — and when a model quietly misses the NPU, nothing errors.
> It just runs on the CPU and drains the battery."

Screen: the README's "Results at a glance" table.

## 0:30 — Catch the failure before the hardware does (75s)

```bash
python -m scanner --input models/whisper-tiny-int8w/ --compile-check
```

> "This is a Whisper encoder, quantized the way the documentation says. The
> scanner reads the graph, applies the NPU's precision rules, and then runs
> Qualcomm's own compiler locally. Both agree: nine LayerNorms are rejected, the
> graph gets split into nine fragments. Four seconds, on a PC with no NPU in it."

Point at the verdict line and the op names.

> "The reason is one sign bit, on one input, of one op type. The weights were
> signed int8 and the Hexagon NPU refuses that for LayerNorm's gamma."

## 1:45 — And the chip agrees (45s)

Open `models/reports/device_logs/j57edm49p_FAILED_int8-weights.log`, search for
`3110`.

> "This is the real Snapdragon X Elite's own log, from running that exact model.
> Same nine rejections the scanner predicted. This file is committed in the repo —
> you don't have to take my word for any number in it."

## 2:30 — The fixed model (45s)

```bash
python -m scanner --input models/whisper-tiny-qdq/ --compile-check
```

> "Unsigned weights, 16-bit activations, real-speech calibration. 100%, one NPU
> graph. On the device: 185 of 187 layers on the NPU, 17.8 ms median — and the
> word error rate is identical to the unquantized model."

## 3:15 — It works (60s)

```bash
python -m server.app
```

Open http://127.0.0.1:8000, transcribe a sample clip.

> "Real transcription, on-device, with the reference text and word error rate
> beside it. And it reports the execution provider that actually attached — right
> now that says CPU, because this machine has no NPU. It would be trivial to
> print 'NPU' here. That lie is exactly what this project exists to catch."

## 4:15 — It isn't one model (45s)

Open `models/reports/zoo/README.md`.

> "Five more models through the same pipeline, each profiled on a real X Elite.
> MobileNetV2 is the one to look at: quantized the naive way it places perfectly
> on the NPU — one graph, nothing on CPU — and classifies seven percent of images
> correctly. Every placement check passes. Only an accuracy check catches it.
> Through this pipeline, it's back to 79.5%, matching the original."

## 5:00 — Close (15s)

> "One command on a Snapdragon laptop tells you whether your model really runs
> on the NPU, which ops don't, and why. MIT licensed, every claim traceable to a
> committed file."

---

## If you have a Snapdragon laptop to record on

Lead with this instead of the setup above:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup-snapdragon.ps1
```

> "ARM64 Python: checked. NPU visible to ONNX Runtime: checked. And then the part
> that matters — it builds a session with CPU fallback disabled. If a single node
> would fall back, this refuses to start. It didn't. Every layer is on the NPU."

Then run the dashboard and show the provider line reading `QNNExecutionProvider`.
