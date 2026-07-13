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
    1|normal|mnist) mode="normal" ;;
    2|showcase|spectrum|layers) mode="spectrum" ;;
    quick) quick="1" ;;
    *) rest+=("$a") ;;
  esac
done

# No mode on CLI → interactive menu
if [[ -z "$mode" ]]; then
  echo "What do you want to run?"
  echo
  echo "  1) normal     — MNIST Neural Fountain (specialize → LT → Master)"
  echo "  2) layers     — full spectrum: all layer types × all 21 dtypes"
  echo "                 (timing log + CSV under logs/)"
  echo
  read -r -p "Choose [1/2]: " choice
  case "${choice:-}" in
    1|normal|mnist|n|N) mode="normal" ;;
    2|layers|showcase|spectrum|l|L) mode="spectrum" ;;
    *)
      echo "invalid choice: ${choice:-<empty>} (expected 1 or 2)" >&2
      exit 2
      ;;
  esac
  echo

  if [[ "$mode" == "normal" && -z "$quick" ]]; then
    read -r -p "Quick smoke (4k subset · K=8)? [y/N]: " q
    case "${q:-}" in y|Y|yes|YES) quick="1" ;; esac
  fi
  if [[ "$mode" == "spectrum" && -z "$quick" ]]; then
    read -r -p "Full spectrum (all 21 dtypes), or quick subset? [F/q]: " q
    case "${q:-}" in q|Q|quick) quick="1" ;; esac
  fi
  echo
fi

case "$mode" in
  normal)
    if [[ -n "$quick" ]]; then
      echo "→ normal · quick MNIST"
      go run . -quick "${rest[@]}"
    else
      echo "→ normal · full MNIST"
      go run . "${rest[@]}"
    fi
    ;;
  spectrum)
    if [[ -n "$quick" ]]; then
      echo "→ layers · spectrum quick (10 families × 7 dtypes)"
      go run . showcase -quick "${rest[@]}"
    else
      echo "→ layers · FULL spectrum (micro-specialize → LT → Master · ≥90% gate · log+csv)"
      echo "  (this trains for real — expect minutes, not sub-second smoke)"
      go run . showcase "${rest[@]}"
    fi
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac

echo
echo "tips:"
echo "  ./run.sh                 # interactive: normal vs layers"
echo "  ./run.sh 1               # MNIST Master"
echo "  ./run.sh 2               # full layer×dtype spectrum + logs/"
echo "  ./run.sh 1 quick         # MNIST smoke"
echo "  ./run.sh 2 quick         # spectrum smoke"
echo "  ./run.sh layers -family dense,mha -dtype float32,int8"
