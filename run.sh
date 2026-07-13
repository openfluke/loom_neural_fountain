#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  loom_neural_fountain — specialists · LT · Master (no seeds) ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

if [[ "${1:-}" == "quick" ]]; then
  shift
  go run . -quick "$@"
else
  go run . "$@"
fi

echo
echo "tips:"
echo "  ./run.sh                 # full 80/20 train · K=16 specialists"
echo "  ./run.sh quick           # 4k train subset · K=8"
echo "  go run . -k 32 -epochs 8"
