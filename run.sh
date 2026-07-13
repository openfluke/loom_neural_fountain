#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  loom_neural_fountain — specialists · LT · Master (no seeds) ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

mode=""
quick=""
rest=()
for a in "$@"; do
  case "$a" in
    showcase|spectrum) mode="showcase" ;;
    quick) quick="quick" ;;
    *) rest+=("$a") ;;
  esac
done

if [[ "$mode" == "showcase" ]]; then
  if [[ -n "$quick" ]]; then
    go run . showcase -quick "${rest[@]}"
  else
    go run . showcase "${rest[@]}"
  fi
elif [[ -n "$quick" ]]; then
  go run . -quick "${rest[@]}"
else
  go run . "${rest[@]}"
fi

echo
echo "tips:"
echo "  ./run.sh                      # full MNIST · K=16 specialists"
echo "  ./run.sh quick                # 4k train subset · K=8"
echo "  ./run.sh showcase             # all layers × all 21 dtypes + log/csv"
echo "  ./run.sh showcase quick       # smoke subset of spectrum"
echo "  ./run.sh showcase -family dense,mha -dtype float32,int8"
echo "  go run . -k 32 -epochs 8"
