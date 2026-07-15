#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -e ".[activation,dev]"
echo "Install vLLM separately for the server CUDA version, then run:"
echo 'python -m pip install -e ".[generation]"'
