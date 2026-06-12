#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: ./scripts/run_stage.sh <stage> <config>"
  exit 1
fi

python -m "src.training.$1" --config "$2"
