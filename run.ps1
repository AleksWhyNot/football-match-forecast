param([ValidateSet('analyze','train')][string]$Command='analyze')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$runtimePath=Join-Path $PSScriptRoot '.runtime/tmp'
New-Item -ItemType Directory -Force $runtimePath | Out-Null
$env:TEMP=$runtimePath
$env:TMP=$runtimePath
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONIOENCODING='utf-8'
$pythonPath=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$existingPath=Join-Path (Split-Path $PSScriptRoot -Parent) 'football-season-forecast/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    if (Test-Path -LiteralPath $existingPath) { $pythonPath=$existingPath }
    else {
        py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required' }
        & $pythonPath -m pip install --no-cache-dir -e '.[dev]'
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    }
}
$env:PYTHONPATH=Join-Path $PSScriptRoot 'src'
& $pythonPath -m match_forecast $Command
exit $LASTEXITCODE
