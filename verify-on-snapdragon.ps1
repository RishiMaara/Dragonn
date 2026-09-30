<#
    Hexagon Bridge - verify on a Snapdragon PC, in one command.

    From cmd:

        powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/RishiMaara/Dragonn/main/verify-on-snapdragon.ps1 -OutFile $env:TEMP\verify.ps1; & $env:TEMP\verify.ps1"

    or, if you already have the repository:

        powershell -ExecutionPolicy Bypass -File .\verify-on-snapdragon.ps1

    It fetches the project if it isn't already here, installs what it needs,
    then answers four questions about THIS laptop and writes the answers to a
    file you can send on:

        1. Is the Hexagon NPU reachable from ONNX Runtime at all?
        2. Does Qualcomm's compiler build the shipped model into one NPU graph?
        3. Will ONNX Runtime run it with CPU fallback disabled - every node on
           the NPU, or refuse to start?
        4. How fast is it, cold and warm?

    Nothing is inferred. Every line it prints is something it just measured, and
    a failure is reported as a failure rather than skipped.

    Expect 3-6 minutes on the first run (most of it pip), under 1 minute after.
#>

[CmdletBinding()]
param(
    [string]$Branch = "main",
    [switch]$SkipInstall,                      # dependencies already present
    [switch]$AllowNonArm64,                    # run the compile-only checks on an x64 PC
    [int]$Runs = 100,                          # latency samples
    [string]$OutDir = $PSScriptRoot
)

$ErrorActionPreference = "Stop"
$started = Get-Date
$REPO = "https://github.com/RishiMaara/Dragonn"

function Head($t) { Write-Host "`n=== $t" -ForegroundColor Cyan }
function Good($t) { Write-Host "  OK   $t" -ForegroundColor Green }
function Bad($t)  { Write-Host "  FAIL $t" -ForegroundColor Red }
function Note($t) { Write-Host "       $t" -ForegroundColor DarkGray }

$result = [ordered]@{
    tool = "Hexagon Bridge verify-on-snapdragon"
    started = $started.ToString("yyyy-MM-dd HH:mm:ss zzz")
    machine = [ordered]@{}
    npu = [ordered]@{}
    compile_check = [ordered]@{}
    strict_session = [ordered]@{}
    latency = [ordered]@{}
    stages_seconds = [ordered]@{}
}
function Stage($name, $block) {
    $t = Get-Date
    & $block
    $result.stages_seconds[$name] = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
}

# -- 0. Where is the project? -------------------------------------------------
Head "Project"
$root = $PSScriptRoot
if (-not (Test-Path (Join-Path $root "scanner"))) {
    $root = Join-Path $env:TEMP "hexagon-bridge"
    if (Test-Path (Join-Path $root "scanner")) {
        Good "Using the copy already downloaded to $root"
    } else {
        Note "Downloading the project (about 15 MB) ..."
        Stage "download" {
            $zip = Join-Path $env:TEMP "hexagon-bridge.zip"
            Invoke-WebRequest "$REPO/archive/refs/heads/$Branch.zip" -OutFile $zip -UseBasicParsing
            $expand = Join-Path $env:TEMP "hexagon-bridge-unzip"
            if (Test-Path $expand) { Remove-Item -LiteralPath $expand -Recurse -Force }
            Expand-Archive -LiteralPath $zip -DestinationPath $expand -Force
            $inner = Get-ChildItem $expand -Directory | Select-Object -First 1
            if (Test-Path $root) { Remove-Item -LiteralPath $root -Recurse -Force }
            Move-Item -LiteralPath $inner.FullName -Destination $root
        }
        Good "Downloaded to $root"
    }
    if (-not $PSBoundParameters.ContainsKey("OutDir") -or -not $OutDir) { $OutDir = $root }
}
if (-not $OutDir) { $OutDir = $root }
Set-Location $root
Good "Project at $root"

# -- 1. The machine -----------------------------------------------------------
# QnnHtp.dll is an ARM64 binary. Under x64 Python (Prism emulation) it cannot
# load, QNN EP silently never appears, and everything below would quietly
# measure a CPU. So this is checked first and hard.
Head "This machine"
$arch = python -c "import platform; print(platform.machine())" 2>$null
$pyver = python -c "import platform; print(platform.python_version())" 2>$null
if (-not $arch) {
    Bad "No python on PATH. Install ARM64 Python 3.11-3.13 from python.org and rerun."
    exit 1
}
$result.machine.python_version = $pyver
$result.machine.python_machine = $arch
$result.machine.processor = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
$result.machine.os = (Get-CimInstance Win32_OperatingSystem).Caption
$result.machine.computer_model = (Get-CimInstance Win32_ComputerSystem).Model
Note "$($result.machine.processor)  |  $($result.machine.computer_model)"
if ($arch -ne "ARM64") {
    if (-not $AllowNonArm64) {
        Bad "Python reports $arch, not ARM64 - this is x64 Python under emulation."
        Note "The Hexagon NPU is unreachable from it. Install ARM64 Python and rerun."
        Note "To run the compile-only checks here anyway: -AllowNonArm64"
        $result.machine.verdict = "not ARM64 python"
        ($result | ConvertTo-Json -Depth 6) | Out-File (Join-Path $OutDir "snapdragon-results.json") -Encoding utf8
        exit 1
    }
    Bad "Python reports $arch, not ARM64 - continuing because -AllowNonArm64 was given."
    Note "There is no NPU to execute on here. The compiler checks below are real;"
    Note "any latency number would not be, so none is measured."
    $result.machine.verdict = "x64 - compile-only, not an NPU run"
} else {
    Good "ARM64-native Python $pyver"
    $result.machine.verdict = "ARM64"
}

# -- 2. Dependencies ----------------------------------------------------------
if (-not $SkipInstall) {
    Head "Installing (first run only, a few minutes)"
    Stage "install" {
        python -m pip install --quiet --upgrade pip
        python -m pip install --quiet onnxruntime numpy onnx rich
        python -m pip install --quiet "onnxruntime-qnn>=2.6"
    }
    if ($LASTEXITCODE -ne 0) { Bad "pip failed - see the output above"; exit 1 }
    Good "onnxruntime, onnxruntime-qnn, onnx, numpy, rich"
} else {
    Head "Skipping install (-SkipInstall)"
}

# -- 3-5. The measurements ----------------------------------------------------
# Done in Python, because that is where the project's own code lives: the same
# functions the cloud results came from, run here instead.
$py = @'
import json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, ".")
out = {}
model = Path("models/whisper-tiny-qdq/encoder_model.onnx")

# --- is the NPU visible at all -------------------------------------------
import onnxruntime as ort
from runtime.qnn_ep import compile_only_available, create_session, npu_available, register_plugin

out["ep"] = {"registered": bool(register_plugin()), "onnxruntime": ort.__version__}
try:
    import onnxruntime_qnn
    out["ep"]["onnxruntime_qnn"] = onnxruntime_qnn.__version__
except Exception:
    out["ep"]["onnxruntime_qnn"] = None
out["ep"]["devices"] = [
    {"ep": d.ep_name, "type": str(d.device.type), "vendor": d.device.vendor}
    for d in (ort.get_ep_devices() if hasattr(ort, "get_ep_devices") else [])
    if d.ep_name == "QNNExecutionProvider"
]
out["ep"]["npu_available"] = bool(npu_available())
out["ep"]["compile_only"] = bool(compile_only_available())
print("NPU_AVAILABLE=" + str(out["ep"]["npu_available"]))

# --- what does Qualcomm's compiler do with the shipped model --------------
from scanner.htp_compile import compile_check

t = time.time()
out["compile_check"] = compile_check(str(model))
out["compile_check"]["wall_seconds"] = round(time.time() - t, 1)
print("COMPILE_OK=" + str(out["compile_check"].get("ok")))
print("NPU_GRAPHS=" + str(out["compile_check"].get("npu_graphs")))
print("CPU_OPS=" + json.dumps(out["compile_check"].get("cpu_ops") or {}))

# --- will ORT run it with CPU fallback disabled ---------------------------
# The whole claim in one call: if any node would land on CPU, this raises.
cache = Path(".qnn_cache_verify")
strict = {"attempted": True}
t = time.time()
try:
    session = create_session(model, {"htp_arch": "73"}, cache_dir=cache, strict=True)
    strict.update(passed=True, providers=session.get_providers(),
                  cold_load_seconds=round(time.time() - t, 2))
except Exception as exc:
    strict.update(passed=False, error=str(exc).splitlines()[0][:300])
out["strict_session"] = strict
print("STRICT_PASSED=" + str(strict.get("passed")))

# --- how fast, cold and warm ---------------------------------------------
if strict.get("passed") and out["ep"]["npu_available"]:
    import numpy as np

    t = time.time()
    session = create_session(model, {"htp_arch": "73"}, cache_dir=cache, strict=True)
    warm_load = round(time.time() - t, 2)

    name = session.get_inputs()[0].name
    shape = [d if isinstance(d, int) else 1 for d in session.get_inputs()[0].shape]
    feed = {name: np.random.randn(*shape).astype(np.float32)}

    for _ in range(5):
        session.run(None, feed)                      # warm the graph
    times = []
    for _ in range(RUNS):
        s = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - s) * 1000)
    times.sort()
    out["latency"] = {
        "runs": len(times),
        "median_ms": round(statistics.median(times), 2),
        "p90_ms": round(times[int(len(times) * 0.9) - 1], 2),
        "min_ms": round(times[0], 2),
        "cold_load_seconds": strict.get("cold_load_seconds"),
        "warm_load_seconds": warm_load,
        "providers": session.get_providers(),
    }
    print("MEDIAN_MS=" + str(out["latency"]["median_ms"]))
    print("WARM_LOAD=" + str(warm_load))

else:
    out["latency"] = {"skipped": "no NPU on this machine - a latency number here would be a CPU number"}
    print("LATENCY_SKIPPED=1")

Path("verify-python-output.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
'@
$py = "RUNS = $Runs`n" + $py
$pyFile = Join-Path $env:TEMP "hexbridge_verify.py"
$py | Out-File -FilePath $pyFile -Encoding utf8

Head "Measuring on this laptop"
Stage "measure" { python $pyFile }
if ($LASTEXITCODE -ne 0) { Bad "The measurement script failed - output above"; exit 1 }

$measured = Get-Content (Join-Path $root "verify-python-output.json") -Raw | ConvertFrom-Json
$result.npu = $measured.ep
$result.compile_check = $measured.compile_check
$result.strict_session = $measured.strict_session
$result.latency = $measured.latency

# -- Report -------------------------------------------------------------------
Head "Results"
if ($measured.ep.npu_available) { Good "Hexagon NPU visible to ONNX Runtime (QNN EP $($measured.ep.onnxruntime_qnn))" }
else { Bad "No NPU device exposed by QNN EP on this machine" }

if ($measured.compile_check.ok) {
    Good "Qualcomm's compiler: ONE NPU graph, nothing left on CPU ($($measured.compile_check.compile_s) s)"
} else {
    Bad "Compiler split the graph: $($measured.compile_check.npu_graphs) NPU graphs; CPU ops: $($measured.compile_check.cpu_ops | ConvertTo-Json -Compress)"
}

if ($measured.strict_session.passed -and $measured.ep.npu_available) {
    Good "ONNX Runtime built the session with CPU fallback DISABLED - every node ran on the NPU"
} elseif ($measured.strict_session.passed) {
    Good "QNN EP accepted every node with CPU fallback DISABLED (compiled here, executed nowhere - no NPU on this machine)"
} else {
    Bad "Strict session refused: $($measured.strict_session.error)"
}

if ($measured.latency -and $measured.latency.median_ms) {
    Good "Latency: $($measured.latency.median_ms) ms median over $($measured.latency.runs) runs (p90 $($measured.latency.p90_ms) ms)"
    Note "Load: $($measured.latency.cold_load_seconds) s cold, $($measured.latency.warm_load_seconds) s warm (compiled graph cached)"
    Note "Cloud X Elite measured 17.8 ms median for this same model - compare yours above."
} elseif ($measured.latency.skipped) {
    Note "Latency not measured: $($measured.latency.skipped)"
}

$result.total_seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
$json = Join-Path $OutDir "snapdragon-results.json"
$txt  = Join-Path $OutDir "snapdragon-results.txt"
($result | ConvertTo-Json -Depth 6) | Out-File -FilePath $json -Encoding utf8

@"
Hexagon Bridge - measured on this PC
$($result.started)

Machine        : $($result.machine.processor)
Model          : $($result.machine.computer_model)
OS             : $($result.machine.os)
Python         : $($result.machine.python_version) ($($result.machine.python_machine))
ONNX Runtime   : $($measured.ep.onnxruntime)   QNN plugin: $($measured.ep.onnxruntime_qnn)

NPU visible    : $($measured.ep.npu_available)
Compiler       : ok=$($measured.compile_check.ok)  npu_graphs=$($measured.compile_check.npu_graphs)  cpu_ops=$($measured.compile_check.cpu_ops | ConvertTo-Json -Compress)  qnn=$($measured.compile_check.qnn_version)
Strict session : passed=$($measured.strict_session.passed)  providers=$($measured.strict_session.providers -join ',')
Latency        : $(if ($measured.latency.median_ms) { "median $($measured.latency.median_ms) ms, p90 $($measured.latency.p90_ms) ms, min $($measured.latency.min_ms) ms over $($measured.latency.runs) runs" } else { "not measured - $($measured.latency.skipped)" })
Load           : $(if ($measured.latency.median_ms) { "$($measured.latency.cold_load_seconds) s cold / $($measured.latency.warm_load_seconds) s warm" } else { "n/a" })
Title line     : $(if ($measured.ep.npu_available) { "this IS an NPU run" } else { "compile-only run, no NPU present" })
Total runtime  : $($result.total_seconds) s

Model under test: models/whisper-tiny-qdq/encoder_model.onnx (the same file the
cloud results were measured on, committed in the repository).
"@ | Out-File -FilePath $txt -Encoding utf8

Head "Done in $($result.total_seconds) seconds"
Note "Wrote $txt"
Note "Wrote $json"
Note "Send those two files back - they are the record of this run."
