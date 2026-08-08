$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$environmentRoot = Split-Path -Parent $repoRoot
$venvRoot = Join-Path $environmentRoot "venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$pythonExe = $null

if (Test-Path -LiteralPath $venvPython) {
    try {
        & $venvPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonExe = $venvPython
        }
    }
    catch {
        $pythonExe = $null
    }
}

if (-not $pythonExe) {
    $managedPythonRoot = Join-Path $repoRoot ".uv-python"
    $pythonExe = Get-ChildItem -LiteralPath $managedPythonRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName "python.exe" } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}

if (-not $pythonExe) {
    throw "No compatible Python runtime found. Repair $venvRoot or install Python 3.12 under $repoRoot\.uv-python."
}

$env:PYTHONPATH = "$repoRoot;$venvRoot\Lib\site-packages"
& $pythonExe (Join-Path $PSScriptRoot "benchmark_core40.py") @args
exit $LASTEXITCODE
