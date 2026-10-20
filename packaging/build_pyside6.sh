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

# Nuitka's early DLL scan derives "<output-dir>/<input>.build" and needs that
# tree to exist before its own source-directory creation step. Recreate it.
# pyside6-deploy also copies the final artifact into "exec_directory" without
# creating the path, so the destination folder must exist too.
mkdir -p src/linak/deployment dist/pyside6

uv sync --dev --locked
uv run pyside6-deploy -c pysidedeploy.spec --mode onefile --no-install --no-warn

if [[ "$(uname -s)" == "Darwin" ]]; then
    output="dist/pyside6/LiNaK.app"
else
    output="dist/pyside6/LiNaK.bin"
fi

if [[ ! -e "$output" ]]; then
    echo "Build completed but the expected output was not found: $output" >&2
    exit 1
fi

echo "Build complete: $output"