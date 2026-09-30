<#
    Dragonn - one command, on a machine with nothing installed.

        powershell -ExecutionPolicy Bypass -Command "iwr -useb https://raw.githubusercontent.com/RishiMaara/Dragonn/main/verify-on-snapdragon.ps1 -OutFile $env:TEMP\verify.ps1; & $env:TEMP\verify.ps1"

    Installs what is missing (Python included), fetches the project, then runs
    the full device report and prints all of it to the screen: environment,
    static scan, Qualcomm's compiler, a session with CPU fallback disabled,
    encoder latency, the graph cache, a decoder on the NPU, and the project's
    own test suite.

    Everything is on screen as it happens. A copy of the whole session is also
    saved next to the results, so the window can be closed and the run still
    read afterwards.

    First run: 10-20 minutes, nearly all of it downloading (Python, the
    runtime, and a 100 MB pre-compiled model). Later runs: about a minute.
    Add -SkipVendor to drop the 100 MB download and the decoder measurement.
#>

[CmdletBinding()]
param(
    [string]$Branch = "main",
    [int]$Runs = 100,                     # latency samples
    [string]$OutDir = "",                 # defaults to your Desktop
    [switch]$SkipVendor,                  # skip the 100 MB decoder bundle
    [switch]$SkipTests,
    [switch]$NoInstallPython,
    [switch]$NoPause,                     # for automation; otherwise the window waits
    [string]$PythonVersion = "3.13.9"     # only used if Python has to be installed
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is far slower with a progress bar
$started = Get-Date
$REPO = "https://github.com/RishiMaara/Dragonn"

function Head($t) { Write-Host "`n==============================================================================" -ForegroundColor Cyan; Write-Host $t -ForegroundColor Cyan; Write-Host "==============================================================================" -ForegroundColor Cyan }
function Good($t) { Write-Host "  [PASS] $t" -ForegroundColor Green }
function Bad($t)  { Write-Host "  [FAIL] $t" -ForegroundColor Red }
function Note($t) { Write-Host "         $t" -ForegroundColor DarkGray }

if (-not $OutDir) {
    $OutDir = [Environment]::GetFolderPath("Desktop")
    if (-not $OutDir) { $OutDir = $env:USERPROFILE }
}
$log = Join-Path $OutDir "snapdragon-session.txt"
$script:pythonLog = Join-Path $OutDir "snapdragon-output.txt"
try { Start-Transcript -Path $log -Force | Out-Null } catch {
    # An unwritable Desktop should not end the run: fall back to TEMP.
    $OutDir = $env:TEMP
    $log = Join-Path $OutDir "snapdragon-session.txt"
    $script:pythonLog = Join-Path $OutDir "snapdragon-output.txt"
    try { Start-Transcript -Path $log -Force | Out-Null } catch {}
}
Remove-Item -LiteralPath $script:pythonLog -ErrorAction SilentlyContinue

Write-Host "`nDragonn - device verification" -ForegroundColor White
Note "started $($started.ToString('yyyy-MM-dd HH:mm:ss'))"
Note "this window shows everything; a copy is being written to $log"

function Finish($code) {
    $secs = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    Head "FINISHED in $secs seconds"
    Note "Session log : $log"
    Note "Output log  : $script:pythonLog"
    if (Test-Path (Join-Path $PWD "device-report.json")) {
        $dest = Join-Path $OutDir "device-report.json"
        try { Copy-Item (Join-Path $PWD "device-report.json") $dest -Force; Note "Measurements: $dest" } catch {}
    }
    Note "Send back the files above, or a photo of this window."
    try { Stop-Transcript | Out-Null } catch {}
    # Never let the window vanish with the output in it. A console opened just
    # for this run closes the moment the script exits, which is how the first
    # attempt disappeared without a trace.
    if (-not $NoPause) {
        Write-Host ""
        Write-Host "Press Enter to close this window ..." -ForegroundColor Yellow
        try { [void](Read-Host) } catch { Start-Sleep -Seconds 60 }
    }
    exit $code
}

trap {
    Bad "unexpected error: $($_.Exception.Message)"
    Note $_.ScriptStackTrace
    Finish 1
}

# -- 1. This machine ---------------------------------------------------------
Head "STEP 1 - This machine"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$isArm = ($cpu.Architecture -eq 12) -or ($env:PROCESSOR_ARCHITECTURE -eq "ARM64")
Note "CPU      : $($cpu.Name)"
Note "Model    : $((Get-CimInstance Win32_ComputerSystem).Model)"
Note "OS       : $((Get-CimInstance Win32_OperatingSystem).Caption)"
Note "Arch     : $(if ($isArm) { 'ARM64' } else { $env:PROCESSOR_ARCHITECTURE })"
if ($isArm) {
    Good "ARM64 machine - the Hexagon NPU should be reachable"
} else {
    Bad "Not an ARM64 machine - there is no Hexagon NPU here"
    Note "The compiler checks still run and are real; latency will not be measured,"
    Note "because a timing taken here would be a CPU timing."
}

# -- 2. Python ---------------------------------------------------------------
# QnnHtp.dll is an ARM64 binary: on a Snapdragon, x64 Python under emulation
# cannot load it and QNN EP silently never appears. onnxruntime-qnn ships
# wheels for CPython 3.11-3.14 only, so the interpreter has to be in range.
Head "STEP 2 - Python"

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
$candidates += Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
$candidates += Get-ChildItem "C:\Python3*\python.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }

$python = $null
foreach ($c in ($candidates | Where-Object { $_ } | Select-Object -Unique)) {
    $info = Test-Python $c
    if ($info) {
        Note "found Python $($info.version) $($info.machine)  $c$(if (-not $info.suitable) { '   <- not usable here' })"
        if ($info.suitable -and -not $python) { $python = $info }
    }
}

if (-not $python -and -not $NoInstallPython) {
    $arch = if ($isArm) { "arm64" } else { "amd64" }
    Note "No usable Python. Installing Python $PythonVersion ($arch), per-user, no admin needed."
    $installer = Join-Path $env:TEMP "python-$PythonVersion-$arch.exe"
    Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-$arch.exe" -OutFile $installer -UseBasicParsing
    Start-Process -FilePath $installer -Wait -ArgumentList @("/quiet","InstallAllUsers=0","PrependPath=1","Include_pip=1","Include_test=0","AssociateFiles=0")
    foreach ($c in (Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending | ForEach-Object { $_.FullName })) {
        $info = Test-Python $c
        if ($info -and $info.suitable) { $python = $info; break }
    }
    if ($python) { Good "Installed Python $($python.version) $($python.machine)" }
}

if (-not $python) {
    Bad "No usable Python and it could not be installed automatically."
    Note "Install Python $PythonVersion from python.org$(if ($isArm) { ' (the ARM64 installer)' }) and run this again."
    Finish 1
}
Good "Using Python $($python.version) $($python.machine)  -  $($python.exe)"

# -- 3. The project ----------------------------------------------------------
Head "STEP 3 - The project"
$root = $PSScriptRoot
if (-not $root -or -not (Test-Path (Join-Path $root "scanner"))) {
    $root = Join-Path $env:TEMP "dragonn"
    if (-not (Test-Path (Join-Path $root "scanner"))) {
        Note "downloading (about 14 MB, includes the model under test) ..."
        $zip = Join-Path $env:TEMP "dragonn.zip"
        Invoke-WebRequest "$REPO/archive/refs/heads/$Branch.zip" -OutFile $zip -UseBasicParsing
        $expand = Join-Path $env:TEMP "dragonn-unzip"
        if (Test-Path $expand) { Remove-Item -LiteralPath $expand -Recurse -Force }
        Expand-Archive -LiteralPath $zip -DestinationPath $expand -Force
        $inner = Get-ChildItem $expand -Directory | Select-Object -First 1
        if (Test-Path $root) { Remove-Item -LiteralPath $root -Recurse -Force }
        Move-Item -LiteralPath $inner.FullName -Destination $root
    }
}
Set-Location $root
Good "Project at $root"

# -- 4. Runtime --------------------------------------------------------------
# A venv, so nothing on this machine is altered and a half-broken system
# Python cannot poison the measurements.
Head "STEP 4 - Runtime (first run only, a few minutes)"
$venv = Join-Path $root ".venv-verify"
$vpy  = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    Note "creating an isolated environment ..."
    & $python.exe -m venv $venv
    if ($LASTEXITCODE -ne 0) { Bad "could not create a virtual environment"; Finish 1 }
}
Note "installing onnxruntime, onnx, numpy, rich, pytest, soundfile ..."
& $vpy -m pip install --quiet --upgrade pip
& $vpy -m pip install --quiet onnxruntime numpy onnx rich pytest soundfile
if ($LASTEXITCODE -ne 0) { Bad "pip could not install the base packages"; Finish 1 }
Note "installing the Hexagon runtime (onnxruntime-qnn) ..."
& $vpy -m pip install --quiet "onnxruntime-qnn>=2.6"
if ($LASTEXITCODE -ne 0) {
    Bad "the QNN plugin would not install for Python $($python.version) $($python.machine)"
    Note "It publishes wheels for CPython 3.11-3.14 on win_arm64 and win_amd64."
    Finish 1
}
Good "runtime installed"

# -- 5. The report ------------------------------------------------------------
Head "STEP 5 - Measuring (everything below is printed as it happens)"
if (-not $SkipVendor) { Note "includes a 100 MB download for the decoder measurement; -SkipVendor omits it" }
$argsList = @("-m", "tools.device_report", "--runs", "$Runs")
if ($SkipVendor) { $argsList += "--skip-vendor" }
if ($SkipTests)  { $argsList += "--skip-tests" }
# ORT 1.30 writes C++ warnings to stderr (e.g. duplicate config keys);
# PowerShell treats those as terminating errors under Stop. Relax it here;
# the exit code is still checked below.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $vpy @argsList 2>&1 | Tee-Object -FilePath $script:pythonLog -Append
$code = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
if ($code -ne 0) { Bad "the report exited with code $code - the output above says where it stopped" }
Finish $code
