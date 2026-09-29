<#
    Hexagon Bridge — one-command setup for a Snapdragon X PC (HP OmniBook and
    every other Windows-on-ARM laptop with a Hexagon NPU).

        powershell -ExecutionPolicy Bypass -File .\setup-snapdragon.ps1

    It installs the device dependencies, then proves the NPU is actually
    reachable before you trust a single number. Every check prints what it
    found, not what it hoped for.
#>

[CmdletBinding()]
param(
    [switch]$SkipInstall,     # dependencies already installed
    [switch]$Serve            # start the dashboard when the checks pass
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Step($text) { Write-Host "`n=== $text" -ForegroundColor Cyan }
function Good($text) { Write-Host "  OK   $text" -ForegroundColor Green }
function Bad($text)  { Write-Host "  FAIL $text" -ForegroundColor Red }
function Note($text) { Write-Host "       $text" -ForegroundColor DarkGray }

# --- 1. ARM64-native Python ---------------------------------------------------
# QnnHtp.dll is an ARM64 binary. Under x64 Python (Prism emulation) it never
# loads, and QNN EP then quietly does not appear — the failure this project
# exists to catch.
Step "Python"
$machine = python -c "import platform; print(platform.machine())"
$version = python -c "import platform; print(platform.python_version())"
if ($machine -ne "ARM64") {
    Bad "Python reports $machine, not ARM64."
    Note "You are on x64 Python under emulation; the Hexagon NPU is unreachable from it."
    Note "Install ARM64 Python from python.org (Windows installer, ARM64) and rerun."
    exit 1
}
Good "ARM64-native Python $version"

# --- 2. Dependencies ----------------------------------------------------------
if (-not $SkipInstall) {
    Step "Installing device dependencies"
    python -m pip install --upgrade pip --quiet
    python -m pip install -r requirements-device.txt
    if ($LASTEXITCODE -ne 0) { Bad "pip install failed"; exit 1 }
    Good "requirements-device.txt installed"
} else {
    Step "Skipping install (-SkipInstall)"
}

# --- 3. Is the NPU actually there? -------------------------------------------
# onnxruntime-qnn 2.x is a plugin EP: it is invisible until registered, and the
# classic providers=[...] argument silently yields a CPU session. scripts/qnn_ep
# registers it properly and reports the device types it exposes.
Step "Hexagon NPU"
$probe = @'
import json, platform
from runtime.qnn_ep import register_plugin, npu_available, _plugin_devices
registered = register_plugin()
devices = [(d.ep_name, str(d.device.type), d.device.vendor) for d in _plugin_devices()]
print(json.dumps({"registered": registered, "npu": npu_available(),
                  "devices": devices, "processor": platform.processor()}))
'@
$probe | Out-File -FilePath "$env:TEMP\hexbridge_probe.py" -Encoding utf8
$json = python "$env:TEMP\hexbridge_probe.py" | Select-Object -Last 1
$info = $json | ConvertFrom-Json

if (-not $info.registered) {
    Bad "QNN execution provider is not installed."
    Note "python -m pip install onnxruntime-qnn"
    exit 1
}
if (-not $info.npu) {
    Bad "QNN EP loaded, but exposes no NPU device on this machine."
    Note "Devices seen: $($info.devices | ForEach-Object { $_ -join '/' })"
    Note "On a Snapdragon X PC, update the Hexagon NPU driver (30.0.140.0 or newer)."
    Note "Windows Update -> Optional updates, or HP Support Assistant on an HP laptop."
    exit 1
}
Good "Hexagon NPU visible to ONNX Runtime — $($info.processor)"

# --- 4. Does the shipped model really land on it? ----------------------------
# Static analysis, then Qualcomm's own compiler, then ONNX Runtime's strict
# no-CPU-fallback session: three independent answers to the same question.
Step "Shipped model on the NPU"
$model = "models\whisper-tiny-qdq\encoder_model.onnx"
if (-not (Test-Path $model)) {
    Note "No converted model yet. Build one on a prep host:"
    Note "  python -m tools.run_pipeline --model openai/whisper-tiny --quantized-dir ./models/whisper-tiny-qdq"
    Note "Skipping the model check."
} else {
    python -m scanner --input models\whisper-tiny-qdq\ --compile-check
    $strict = @'
import sys
from pathlib import Path
from runtime.qnn_ep import create_session
try:
    session = create_session(Path(r"models/whisper-tiny-qdq/encoder_model.onnx"),
                             {"htp_arch": "73"}, cache_dir=Path(".qnn_cache"), strict=True)
    print("STRICT_OK " + ",".join(session.get_providers()))
except Exception as exc:
    print("STRICT_FAIL " + str(exc).splitlines()[0])
'@
    $strict | Out-File -FilePath "$env:TEMP\hexbridge_strict.py" -Encoding utf8
    $result = python "$env:TEMP\hexbridge_strict.py" | Select-Object -Last 1
    if ($result -like "STRICT_OK*") {
        Good "Session built with CPU fallback disabled — every node is on the NPU"
        Note $result
    } else {
        Bad "ONNX Runtime refused an NPU-only session for this model"
        Note $result
        Note "The scanner output above names the ops that would fall back."
    }
}

# --- 5. Run it ----------------------------------------------------------------
Step "Ready"
Note "Dashboard:      python -m server.app      then open http://127.0.0.1:8000"
Note "Scan any model: python -m scanner --input <model.onnx> --compile-check"
Note "Accuracy:       python -m tools.eval_wer"

if ($Serve) {
    Step "Starting the dashboard on http://127.0.0.1:8000"
    python -m server.app
}
