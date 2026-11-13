[CmdletBinding()]
param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "This script must run on Windows. Use a Windows machine or windows-latest CI runner."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found on PATH. Install uv before building Qali."
}

if ($Clean) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
        (Join-Path $repoRoot "dist/pyside6"),
        (Join-Path $repoRoot "src/qali/deployment")
}

# Nuitka's early DLL scan derives "<output-dir>/<input>.build" and needs that
# tree to exist before its own source-directory creation step. Recreate it.
# pyside6-deploy also copies the final exe into "exec_directory" without
# creating the path, so dist/pyside6 must exist too.
New-Item -ItemType Directory -Force `
    (Join-Path $repoRoot "src/qali/deployment"),
    (Join-Path $repoRoot "dist/pyside6") | Out-Null

# Nuitka 4.1.1 officially supports Python up to 3.13, so pin the build
# interpreter instead of reusing whatever .venv exists.
$python = "3.13"

Write-Host "Syncing development environment..."
uv sync --dev --locked --python $python
if ($LASTEXITCODE -ne 0) {
    throw "uv sync failed with exit code $LASTEXITCODE."
}

Write-Host "Building with pyside6-deploy..."
uv run --python $python pyside6-deploy -c pysidedeploy.spec --mode onefile --no-install --no-warn
if ($LASTEXITCODE -ne 0) {
    throw "pyside6-deploy failed with exit code $LASTEXITCODE."
}

$output = Join-Path $repoRoot "dist/pyside6/Qali.exe"
if (-not (Test-Path $output)) {
    throw "Build completed but the expected output was not found: $output"
}

Write-Host "Build complete: $output"