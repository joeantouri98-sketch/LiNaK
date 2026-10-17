#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" && "$(uname -s)" != "Linux" ]]; then
    echo "This script must run on macOS or Linux." >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found on PATH. Install uv before building LiNaK." >&2
    exit 1
fi

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf dist/pyside6 src/linak/deployment
elif [[ "${1:-}" != "" ]]; then
    echo "Usage: $0 [--clean]" >&2
    exit 2
fi

uv sync --dev
uv run pyside6-deploy -c pysidedeploy.spec --mode onefile --no-install --no-warn

if [[ "$(uname -s)" == "Darwin" ]]; then
    output="dist/pyside6/LiNaK.app"
elif [[ "$(uname -s)" == "Linux" ]]; then
    output="dist/pyside6/LiNaK.bin"
else
    output="dist/pyside6/LiNaK"
fi

if [[ ! -e "$output" ]]; then
    echo "Build completed but the expected output was not found: $output" >&2
    exit 1
fi

echo "Build complete: $output"