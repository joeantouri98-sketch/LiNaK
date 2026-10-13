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
        (Join-Path $repoRoot "dist/windows"),
        (Join-Path $repoRoot "build/pyinstaller-windows")
}

Write-Host "Syncing development environment..."
uv sync --dev

$distPath = Join-Path $repoRoot "dist/windows"
$workPath = Join-Path $repoRoot "build/pyinstaller-windows"
$specPath = Join-Path $repoRoot "packaging/LiNaK-windows.spec"

$pyInstallerArgs = @(
    "run", "pyinstaller",
    "--noconfirm",
    "--clean",
    "--distpath", $distPath,
    "--workpath", $workPath,
    $specPath
)

Write-Host "Building windowed executable..."

& uv @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}

$executable = Join-Path $distPath "LiNaK/LiNaK.exe"
if (-not (Test-Path $executable)) {
    throw "Build completed but the expected executable was not found: $executable"
}

Write-Host "Build complete: $executable"
