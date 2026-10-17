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
    throw "uv was not found on PATH. Install uv before building LiNaK."
}

if ($Clean) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
        (Join-Path $repoRoot "dist/pyside6"),
        (Join-Path $repoRoot "src/linak/deployment")
}

Write-Host "Syncing development environment..."
uv sync --dev

Write-Host "Building with pyside6-deploy..."
uv run pyside6-deploy -c pysidedeploy.spec --mode onefile --no-install --no-warn
if ($LASTEXITCODE -ne 0) {
    throw "pyside6-deploy failed with exit code $LASTEXITCODE."
}

$output = Join-Path $repoRoot "dist/pyside6/LiNaK.exe"
if (-not (Test-Path $output)) {
    throw "Build completed but the expected output was not found: $output"
}

Write-Host "Build complete: $output"