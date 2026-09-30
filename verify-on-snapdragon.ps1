<#
    Hexagon Bridge - verify on this PC, in one command.

    From cmd, on a machine with nothing installed:

        powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/RishiMaara/Dragonn/main/verify-on-snapdragon.ps1 -OutFile $env:TEMP\verify.ps1; & $env:TEMP\verify.ps1"

    It fetches the project, finds or installs a suitable Python, builds an
    isolated environment, installs the runtime, and then measures four things
    about this laptop:

        1. Is the Hexagon NPU reachable from ONNX Runtime at all?
        2. Does Qualcomm's compiler build the shipped model into one NPU graph?
        3. Will ONNX Runtime run it with CPU fallback disabled - every node on
           the NPU, or refuse to start?
        4. How fast is it, cold and warm?

    It always writes a report, including when something fails: the failure is
    the result in that case, and guessing would be worse than reporting it.

    On a machine without a Hexagon NPU it still runs the compiler checks, which
    are real there, and reports latency as "not measured" rather than timing a
    CPU and calling it an NPU.

    First run: 4-8 minutes, nearly all of it installing. Later runs: under one.
#>

[CmdletBinding()]
param(
    [string]$Branch = "main",
    [int]$Runs = 100,                     # latency samples
    [string]$OutDir = "",                 # defaults to your Desktop
    [switch]$NoInstallPython,             # never install Python, just use what is here
    [string]$PythonVersion = "3.13.9"     # used only if Python has to be installed
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is far slower with a progress bar
$started = Get-Date
$REPO = "https://github.com/RishiMaara/Dragonn"

function Head($t) { Write-Host "`n=== $t" -ForegroundColor Cyan }
function Good($t) { Write-Host "  OK   $t" -ForegroundColor Green }
function Bad($t)  { Write-Host "  FAIL $t" -ForegroundColor Red }
function Note($t) { Write-Host "       $t" -ForegroundColor DarkGray }

if (-not $OutDir) {
    $OutDir = [Environment]::GetFolderPath("Desktop")
    if (-not $OutDir) { $OutDir = $env:USERPROFILE }
}

$result = [ordered]@{
    tool       = "Hexagon Bridge verify-on-snapdragon"
    started    = $started.ToString("yyyy-MM-dd HH:mm:ss zzz")
    machine    = [ordered]@{}
    python     = [ordered]@{}
    npu        = [ordered]@{}
    compile_check  = [ordered]@{}
    strict_session = [ordered]@{}
    latency    = [ordered]@{}
    steps      = [ordered]@{}
    problems   = @()
}

function Save-Report {
    $result.total_seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    $json = Join-Path $OutDir "snapdragon-results.json"
    $txt  = Join-Path $OutDir "snapdragon-results.txt"
    try { ($result | ConvertTo-Json -Depth 8) | Out-File -FilePath $json -Encoding utf8 } catch {}

    $hasNpuAnswer = $result.npu -and ($result.npu.PSObject.Properties.Name -contains "npu_available")
    $npuLine = if ($result.npu.npu_available) { "yes" } elseif ($hasNpuAnswer) { "no" } else { "not reached (run stopped earlier)" }
    $lat = if ($result.latency.median_ms) {
        "median $($result.latency.median_ms) ms, p90 $($result.latency.p90_ms) ms, min $($result.latency.min_ms) ms over $($result.latency.runs) runs"
    } elseif ($result.latency.skipped) { "not measured - $($result.latency.skipped)" } else { "not reached" }

    @"
Hexagon Bridge - measured on this PC
$($result.started)

Machine        : $($result.machine.processor)
Model          : $($result.machine.computer_model)
OS             : $($result.machine.os)
Architecture   : $($result.machine.architecture)
Python         : $($result.python.version) ($($result.python.machine)) at $($result.python.exe)
ONNX Runtime   : $($result.npu.onnxruntime)   QNN plugin: $($result.npu.onnxruntime_qnn)

Run type       : $($result.machine.run_type)
NPU visible    : $npuLine
QNN EP devices : $($result.npu.devices | ForEach-Object { "$($_.ep)/$($_.type)/$($_.vendor)" } | Sort-Object -Unique)
Compiler       : ok=$($result.compile_check.ok)  npu_graphs=$($result.compile_check.npu_graphs)  cpu_ops=$($result.compile_check.cpu_ops | ConvertTo-Json -Compress)  qnn=$($result.compile_check.qnn_version)
Strict session : passed=$($result.strict_session.passed)  $($result.strict_session.error)
Latency        : $lat
Load           : $(if ($result.latency.median_ms) { "$($result.latency.cold_load_seconds) s cold / $($result.latency.warm_load_seconds) s warm" } else { "n/a" })

Problems       : $(if ($result.problems.Count) { ($result.problems -join "; ") } else { "none" })
Total runtime  : $($result.total_seconds) s

Model under test: models/whisper-tiny-qdq/encoder_model.onnx - the same file the
published results were measured on, committed in the repository.
"@ | Out-File -FilePath $txt -Encoding utf8

    Head "Report"
    Note "Wrote $txt"
    Note "Wrote $json"
    Note "Send those two files back - they are the record of this run."
}

trap {
    $result.problems += "unexpected error: $($_.Exception.Message)"
    Bad $_.Exception.Message
    Save-Report
    exit 1
}

# -- 1. This machine ---------------------------------------------------------
Head "This machine"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$isArm = ($cpu.Architecture -eq 12) -or ($env:PROCESSOR_ARCHITECTURE -eq "ARM64")
$result.machine.processor      = $cpu.Name
$result.machine.computer_model = (Get-CimInstance Win32_ComputerSystem).Model
$result.machine.os             = (Get-CimInstance Win32_OperatingSystem).Caption
$result.machine.architecture   = if ($isArm) { "ARM64" } else { $env:PROCESSOR_ARCHITECTURE }
$result.machine.run_type       = if ($isArm) { "NPU run" } else { "compile-only (no Hexagon NPU on this machine)" }
Note "$($cpu.Name)  |  $($result.machine.computer_model)"
if ($isArm) { Good "ARM64 machine - the Hexagon NPU should be reachable" }
else {
    Note "Not an ARM64 machine. The compiler checks below are still real;"
    Note "latency is not measured here, because it would be a CPU number."
}

# -- 2. A Python that can load the Hexagon runtime ---------------------------
# QnnHtp.dll is an ARM64 binary: on a Snapdragon, x64 Python under emulation
# cannot load it and QNN EP silently never appears. So the interpreter has to
# match the machine, and onnxruntime-qnn publishes wheels for 3.11-3.14 only.
Head "Python"

function Test-Python($exe) {
    try {
        $out = & $exe -c "import platform,sys;print(platform.machine()+' '+str(sys.version_info[0])+'.'+str(sys.version_info[1])+'.'+str(sys.version_info[2]))" 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $null }
        $parts = ($out -split ' ')
        $v = [version]$parts[1]
        return [pscustomobject]@{
            exe = $exe; machine = $parts[0]; version = $parts[1]
            suitable = ($v.Major -eq 3 -and $v.Minor -ge 11 -and $v.Minor -le 14) -and
                       ((-not $isArm) -or ($parts[0] -eq "ARM64"))
        }
    } catch { return $null }
}

$candidates = @()
foreach ($c in @("python", "python3")) {
    $found = Get-Command $c -ErrorAction SilentlyContinue
    if ($found) { $candidates += $found.Source }
}
try {
    $candidates += (& py -0p 2>$null | Select-String -Pattern "[A-Za-z]:\\[^ ]*python\.exe" -AllMatches |
                    ForEach-Object { $_.Matches.Value })
} catch {}
$candidates += Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue |
               ForEach-Object { $_.FullName }
$candidates += Get-ChildItem "C:\Python3*\python.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }

$python = $null
foreach ($c in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
    $info = Test-Python $c
    if ($info) {
        $tag = if ($info.suitable) { "" } else { "  (not usable here)" }
        Note "found Python $($info.version) $($info.machine) at $c$tag"
        if ($info.suitable -and -not $python) { $python = $info }
    }
}

if (-not $python -and -not $NoInstallPython) {
    $arch = if ($isArm) { "arm64" } else { "amd64" }
    Note "No suitable Python found. Installing Python $PythonVersion ($arch) for this user - a few minutes."
    $t = Get-Date
    $installer = Join-Path $env:TEMP "python-$PythonVersion-$arch.exe"
    Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-$arch.exe" -OutFile $installer -UseBasicParsing
    # Per-user, silent, on PATH, with pip. No admin rights needed.
    Start-Process -FilePath $installer -Wait -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_pip=1", "Include_test=0", "AssociateFiles=0"
    )
    $result.steps["install_python"] = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    $fresh = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue |
             Sort-Object LastWriteTime -Descending | ForEach-Object { $_.FullName }
    foreach ($c in $fresh) {
        $info = Test-Python $c
        if ($info -and $info.suitable) { $python = $info; break }
    }
    if ($python) { Good "Installed Python $($python.version) $($python.machine)" }
}

if (-not $python) {
    $need = if ($isArm) { "3.11-3.14, ARM64-native" } else { "3.11-3.14" }
    $result.problems += "no usable Python (need $need)"
    Bad "No usable Python, and it could not be installed automatically."
    Note "Install Python $PythonVersion from python.org and rerun."
    Save-Report
    exit 1
}
$result.python.exe     = $python.exe
$result.python.version = $python.version
$result.python.machine = $python.machine
Good "Using Python $($python.version) $($python.machine)"

# -- 3. The project ----------------------------------------------------------
Head "Project"
$root = $PSScriptRoot
if (-not $root -or -not (Test-Path (Join-Path $root "scanner"))) {
    $root = Join-Path $env:TEMP "hexagon-bridge"
    if (-not (Test-Path (Join-Path $root "scanner"))) {
        Note "Downloading the project (about 14 MB, includes the model under test) ..."
        $t = Get-Date
        $zip = Join-Path $env:TEMP "hexagon-bridge.zip"
        Invoke-WebRequest "$REPO/archive/refs/heads/$Branch.zip" -OutFile $zip -UseBasicParsing
        $expand = Join-Path $env:TEMP "hexagon-bridge-unzip"
        if (Test-Path $expand) { Remove-Item -LiteralPath $expand -Recurse -Force }
        Expand-Archive -LiteralPath $zip -DestinationPath $expand -Force
        $inner = Get-ChildItem $expand -Directory | Select-Object -First 1
        if (Test-Path $root) { Remove-Item -LiteralPath $root -Recurse -Force }
        Move-Item -LiteralPath $inner.FullName -Destination $root
        $result.steps["download"] = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    }
}
Set-Location $root
Good "Project at $root"

# -- 4. An isolated environment ----------------------------------------------
# A venv rather than the system Python: nothing on this machine is changed, and
# a half-installed system Python cannot poison the result.
Head "Runtime (first run only, a few minutes)"
$venv = Join-Path $root ".venv-verify"
$vpy  = Join-Path $venv "Scripts\python.exe"
$t = Get-Date
if (-not (Test-Path $vpy)) {
    & $python.exe -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "could not create a virtual environment with $($python.exe)" }
}
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet onnxruntime numpy onnx rich
if ($LASTEXITCODE -ne 0) { throw "pip could not install onnxruntime, onnx and numpy" }
& $vpy -m pip install --quiet "onnxruntime-qnn>=2.6"
if ($LASTEXITCODE -ne 0) {
    $result.problems += "onnxruntime-qnn has no wheel for Python $($python.version) $($python.machine)"
    Bad "The QNN plugin would not install for this Python."
    Note "It publishes wheels for CPython 3.11-3.14 on win_arm64 and win_amd64."
    Save-Report
    exit 1
}
$result.steps["install_runtime"] = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
Good "onnxruntime, onnxruntime-qnn, onnx, numpy, rich"

# -- 5. Measure --------------------------------------------------------------
$py = @'
import json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, ".")
out = {}
model = Path("models/whisper-tiny-qdq/encoder_model.onnx")
if not model.exists():
    print("MODEL_MISSING=1")
    Path("verify-python-output.json").write_text(json.dumps({"error": "model missing"}), encoding="utf-8")
    raise SystemExit(2)

import onnxruntime as ort
from runtime.qnn_ep import compile_only_available, create_session, npu_available, register_plugin

ep = {"registered": bool(register_plugin()), "onnxruntime": ort.__version__}
try:
    import onnxruntime_qnn
    ep["onnxruntime_qnn"] = onnxruntime_qnn.__version__
except Exception:
    ep["onnxruntime_qnn"] = None
ep["devices"] = [
    {"ep": d.ep_name, "type": str(d.device.type), "vendor": d.device.vendor}
    for d in (ort.get_ep_devices() if hasattr(ort, "get_ep_devices") else [])
    if d.ep_name == "QNNExecutionProvider"
]
ep["npu_available"] = bool(npu_available())
ep["compile_only"] = bool(compile_only_available())
out["ep"] = ep
print("NPU_AVAILABLE=" + str(ep["npu_available"]))

from scanner.htp_compile import compile_check

t = time.time()
out["compile_check"] = compile_check(str(model))
out["compile_check"]["wall_seconds"] = round(time.time() - t, 1)
print("COMPILE_OK=" + str(out["compile_check"].get("ok")))

cache = Path(".qnn_cache_verify")
strict = {}
t = time.time()
try:
    session = create_session(model, {"htp_arch": "73"}, cache_dir=cache, strict=True)
    strict.update(passed=True, providers=session.get_providers(),
                  cold_load_seconds=round(time.time() - t, 2))
except Exception as exc:
    strict.update(passed=False, error=str(exc).splitlines()[0][:300])
out["strict_session"] = strict
print("STRICT_PASSED=" + str(strict.get("passed")))

if strict.get("passed") and ep["npu_available"]:
    import numpy as np

    t = time.time()
    session = create_session(model, {"htp_arch": "73"}, cache_dir=cache, strict=True)
    warm_load = round(time.time() - t, 2)

    spec = session.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in spec.shape]
    feed = {spec.name: np.random.randn(*shape).astype(np.float32)}
    for _ in range(5):
        session.run(None, feed)
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
else:
    out["latency"] = {"skipped": "no Hexagon NPU on this machine - any timing here would be a CPU timing"}
    print("LATENCY_SKIPPED=1")

Path("verify-python-output.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
'@
$py = "RUNS = $Runs`n" + $py
$pyFile = Join-Path $env:TEMP "hexbridge_verify.py"
$py | Out-File -FilePath $pyFile -Encoding utf8

Head "Measuring"
$t = Get-Date
& $vpy $pyFile
if ($LASTEXITCODE -ne 0) {
    $result.problems += "the measurement script failed (exit $LASTEXITCODE)"
    Bad "Measurement failed - the output above says why."
    Save-Report
    exit 1
}
$result.steps["measure"] = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)

$m = Get-Content (Join-Path $root "verify-python-output.json") -Raw | ConvertFrom-Json
$result.npu = $m.ep
$result.compile_check = $m.compile_check
$result.strict_session = $m.strict_session
$result.latency = $m.latency

# -- 6. What it found --------------------------------------------------------
Head "Results"
if ($m.ep.npu_available) { Good "Hexagon NPU visible to ONNX Runtime (QNN plugin $($m.ep.onnxruntime_qnn))" }
else { Bad "No NPU device exposed by QNN EP on this machine" }

if ($m.compile_check.ok) {
    Good "Qualcomm's compiler: ONE NPU graph, nothing left on CPU ($($m.compile_check.compile_s) s)"
} else {
    Bad "Compiler split the graph: $($m.compile_check.npu_graphs) NPU graphs, CPU ops $($m.compile_check.cpu_ops | ConvertTo-Json -Compress)"
    $result.problems += "compiler did not produce a single NPU graph"
}

if ($m.strict_session.passed -and $m.ep.npu_available) {
    Good "Session built with CPU fallback DISABLED - every node ran on the NPU"
} elseif ($m.strict_session.passed) {
    Good "QNN EP accepted every node with CPU fallback DISABLED (compiled here, not executed - no NPU present)"
} else {
    Bad "Strict session refused: $($m.strict_session.error)"
    $result.problems += "strict session refused"
}

if ($m.latency.median_ms) {
    Good "Latency: $($m.latency.median_ms) ms median over $($m.latency.runs) runs (p90 $($m.latency.p90_ms) ms)"
    Note "Load: $($m.latency.cold_load_seconds) s cold, $($m.latency.warm_load_seconds) s warm (compiled graph cached)"
    Note "The published cloud X Elite figure for this same model is 17.8 ms median."
} else {
    Note "Latency: $($m.latency.skipped)"
}

Save-Report
Head "Done in $($result.total_seconds) seconds"
